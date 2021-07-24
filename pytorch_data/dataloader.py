from typing import Any, Callable, Generic, Iterable, List, Optional, Sequence, TypeVar
import multiprocessing as python_multiprocessing
import itertools
import threading
import queue
from dataclasses import dataclass

import torch
from torch.utils.data import _utils
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch._six import string_classes

from . import IterableDataset, Dataset, Sampler, SequentialSampler, RandomSampler, BatchSampler

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _InfiniteConstantSampler(Sampler):
    """Sampler for `IterableDataset`
    """
    def __init__(self):
        super().__init__()
    
    def __iter__(self):
        while True:
            yield None


r"""Dummy class used to resume the fetching when worker reuse is enabled"""
@dataclass(frozen=True)
class _ResumeIteration(object):
    pass


class DataLoader(Generic[T_co]):
    """
    Args:
        batch_size (int | None). 1
            When `None`, auto-batching is disabled.
        sampler (Iterable | None). None
            Defines the strategy to draw samples from the dataset. Can be any `Iterable`
            with `__len__` implemented.
        batch_sampler (Iterable | None). None
            Like `sampler`, but returns a batch of indices at a time.
        num_workers (int | None)
            The number of subprocesses to use for data loading. `0` means the data will
            be loaded in the main process.
        collate_fn (Callable | None). None
            Merges a list of samples to form a mini-batch of Tensor(s). Used when using
            batched loading from a map-style dataset.
        generator (torch.Generator | None). None
            RNG.
        prefetch_factor (int | None). 2
            Number of samples loaded in advance by each worker. `2` means there will be a
            total of `2 * num_workers` samples prefetched across all workers.
        persistent_workers (bool, optional) None
            If ``True``, the data loader will not shutdown the worker processes after a
            dataset has been consumed once. This allows to maintain the workers `Dataset`
            instances alive.
    """
    dataset: Dataset[T_co]
    batch_size = Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Sampler
    prefetch_factor: int
    _iterator: Optional['_BaseDataLoaderIter']
    __initializeed = False

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context = None,
        generator = None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        self._sanity_check(num_workers, timeout, prefetch_factor, persistent_workers, dataset, shuffle, sampler, batch_sampler, batch_size, drop_last)
        
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
        else:
            self._dataset_kind = _DatasetKind.Map
        self.dataset = dataset  # ⭐
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        if batch_sampler is not None:
            batch_size = None
            drop_last = False
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self._set_default_utils(sampler, batch_sampler, collate_fn, generator, shuffle)

        self.persistent_workers = persistent_workers
        
        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]
    
    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.persistent_workers and self.num_workers > 0:
            # For multiple workers and `persistent_workers is True`, workers are reused
            # so the iterator is created only once in the lifetime of the DataLoader 
            # object.
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            # for single worker, the iterator is created everytime to avoid reseting its state.
            return self._get_iterator()

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # For `IterableDataset`, when multi-processing data loading is done naively, this length is inaccurate.
            # But as long as `__len__` is called, this length will be recorded so that at least some warnings can be raised. 
            length = self._IterableDataset_len_called = len(self.dataset)
            # IterableDataset doesn't allow custom sampler or batch_sampler, so we calculate manually.
            if self.batch_size is not None:
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)
    
    @property
    def _auto_collation(self):
        return self.batch_sampler is not None
    
    @property
    def _index_sampler(self):
        if self.batch_sampler is not None:
            return self.batch_sampler
        else:
            return self.sampler

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context
    
    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        # sanity check for context
        self.__multiprocessing_context = multiprocessing_context
    
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    def check_worker_number_rationality():
        # We don't take threading into account since each worker process is single threaded
        # at this time.
        pass

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, string_classes):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {!r}, but got '
                             'multiprocessing_context={!r}').format(valid_start_methods, multiprocessing_context))
                    # error: Argument 1 to "get_context" has incompatible type "Union[str, bytes]"; expected "str"  [arg-type]
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)  # type: ignore[arg-type]

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError(('multiprocessing_context option should be a valid context '
                                     'object or a string specifying the start method, but got '
                                     'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def _set_default_utils(self, sampler, batch_sampler, collate_fn, generator, shuffle):        
        if sampler is None:  # set default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(self.dataset, generator=generator)
                else:
                    sampler = SequentialSampler(self.dataset)

        if self.batch_size is not None and batch_sampler is None:
            # auto_collation with default batch_sampler
            batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)
        
        if collate_fn is None:
            if self.batch_sampler is not None:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.sampler = sampler  # ⭐
        self.batch_sampler = batch_sampler  # ⭐
        self.collate_fn = collate_fn  # ⭐
        self.generator = generator
        self.shuffle = shuffle

    def _sanity_check(self, num_workers, timeout, prefetch_factor, persistent_workers, dataset, shuffle, sampler, batch_sampler, batch_size, drop_last):
        assert num_workers >= 0
        assert timeout >= 0
        if num_workers == 0:
            assert prefetch_factor == 2
        assert prefetch_factor > 0
        if persistent_workers:
            assert num_workers > 0

        if isinstance(dataset, IterableDataset):
            # Iterable-style datasets are incompatible with custom samplers.
            # This is not only because iterable-style datasets don't use key, but also a
            # design choice for simplicity, especially in multi-process data loading.
            assert shuffle is False
            assert sampler is None
            assert batch_sampler is None
        else:
            self._dataset_kind = _DatasetKind.Map
        
        if shuffle is True:
            assert sampler is None
        
        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            # default values for these parameters
            assert batch_size == 1
            assert shuffle is False
            assert sampler is None
            assert drop_last is False
        elif batch_size is None:  # no auto_collation
            assert drop_last is False


class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader.batch_sampler is not None
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)

    
    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:  # Why could it be None???
                self._reset()
            data = self._next_data()
            self._num_yielded += 1
            # warning here for IterableDataset
            return data


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None:
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last
        )
    
    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data
    

class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue()  # ⭐
        self._worker_pids_set = False
        self._shutdown = False
        # used to signal the workers that the iterator is shutting down so that they will
        # not send processed data to queues anymore and only wait for the final `None`
        # before exiting.
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []  # ⭐
        self._workers = []

        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                )
            )
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
        
        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._data_queue = queue.Queue()  # ⭐
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    torch.cuda.current_device(),
                    self._pin_memory_thread_done_event
                )
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # map: task idx =>  - (worker_id,)          if data isn't fetched (outstanding)
        #                   \ (worker_id, data)     if data is already fetched (out-of-order)             
        self._task_info = {}  # ⭐
        # always equal to count(v for v in task_info.values() if len(v) == 1)
        self._tasks_outstanding = 0
        # A list of booleans representing whether each worker still has work to do. It
        # always contains all `True`s if not using an iterable-style dataset
        self._workers_status = [True] * self._num_workers
        # resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()
    
    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                raise RuntimeError
            if isinstance(e, queue.Empty):
                return (False, None)
            # check file descriptor limit

    def _get_data(self):
        # Fetches data from `self._data_queue`.
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:  # pin_memory_thread is dead.
                raise RuntimeError
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                # has data or is still active
                if len(info) == 2 or self._workers_status[worker_id]:
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch
        
            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue
                
            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
            
    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        assert self._workers_done_event.is_set() == shutdown
    