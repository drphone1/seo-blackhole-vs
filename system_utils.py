# system_utils.py
# Created: 2025-02-03 13:44:02 UTC
# Last modified: 2025-02-03 18:15:03 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import gc
import sys
import psutil
import logging
import threading
import contextlib
from typing import Dict, List, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from config import CONFIG
from exceptions import (
    ResourceError,
    SystemError,
    MemoryError,
    ProcessError,
    ThreadError,
    error_tracker
)

logger = logging.getLogger(__name__)

@dataclass
class SystemStats:
    """System resource statistics."""
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    open_files: int
    thread_count: int
    process_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

class MemoryManager:
    """Advanced memory resource management and monitoring."""
    
    def __init__(
        self,
        threshold: float = CONFIG.resources.max_memory_percent,
        leak_detection: bool = True,
        gc_threshold: Tuple[int, int, int] = CONFIG.resources.gc_threshold,
        min_free: int = CONFIG.resources.min_free_memory
    ):
        self.threshold = threshold
        self.leak_detection = leak_detection
        self.min_free = min_free
        self.process = psutil.Process(os.getpid())
        self._last_check = datetime.utcnow()
        self._last_memory = self.process.memory_percent()
        self._memory_samples: List[Tuple[datetime, float]] = []
        self._lock = threading.Lock()
        
        # Configure garbage collector
        gc.set_threshold(*gc_threshold)
        gc.enable()
    
    def check_memory(self) -> None:
        """Check memory usage and perform cleanup if needed."""
        with self._lock:
            current = self.process.memory_percent()
            system_memory = psutil.virtual_memory()
            
            # Check system memory
            if system_memory.available < self.min_free:
                self._emergency_cleanup()
                raise ResourceError(
                    "System memory critically low",
                    resource_type="memory",
                    current_value=system_memory.available,
                    limit=self.min_free
                )
            
            # Check process memory
            if current > self.threshold:
                self._cleanup()
                if self.process.memory_percent() > self.threshold:
                    raise ResourceError(
                        "Memory usage above threshold",
                        resource_type="memory",
                        current_value=current,
                        limit=self.threshold
                    )
            
            # Leak detection
            if self.leak_detection:
                self._detect_leaks(current)
    
    def _detect_leaks(self, current: float) -> None:
        """Detect potential memory leaks."""
        now = datetime.utcnow()
        self._memory_samples.append((now, current))
        
        # Keep last hour of samples
        cutoff = now - timedelta(hours=1)
        self._memory_samples = [
            (t, m) for t, m in self._memory_samples
            if t > cutoff
        ]
        
        # Check for consistent increase
        if len(self._memory_samples) >= 10:
            times, memories = zip(*self._memory_samples[-10:])
            if all(memories[i] < memories[i+1] for i in range(len(memories)-1)):
                increase = memories[-1] - memories[0]
                logger.warning(
                    f"Possible memory leak detected: {increase:.1f}% increase over "
                    f"{(times[-1] - times[0]).total_seconds():.0f} seconds"
                )
    
    def _cleanup(self) -> None:
        """Perform memory cleanup."""
        gc.collect()
        logger.info("Memory cleanup performed")
    
    def _emergency_cleanup(self) -> None:
        """Perform emergency memory cleanup."""
        gc.collect(2)  # Collect all generations
        self._clear_caches()
        logger.warning("Emergency memory cleanup performed")
    
    def _clear_caches(self) -> None:
        """Clear various Python caches."""
        gc.collect()
        with contextlib.suppress(Exception):
            sys.modules.clear()
            threading._limbo.clear()
    
    def get_stats(self) -> Dict[str, Union[float, int]]:
        """Get detailed memory statistics."""
        process_memory = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_percent': self.process.memory_percent(),
            'process_rss': process_memory.rss,
            'process_vms': process_memory.vms,
            'system_total': system_memory.total,
            'system_available': system_memory.available,
            'system_percent': system_memory.percent
        }

class ProcessManager:
    """Process management and monitoring."""
    
    def __init__(
        self,
        max_cpu_percent: float = CONFIG.resources.max_cpu_percent,
        monitor_interval: int = CONFIG.resources.memory_check_interval
    ):
        self.max_cpu_percent = max_cpu_percent
        self.monitor_interval = monitor_interval
        self.process = psutil.Process(os.getpid())
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        self._stats_history: List[SystemStats] = []
    
    def start_monitoring(self) -> None:
        """Start background process monitoring."""
        if not self._monitor_thread:
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Process monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background process monitoring."""
        if self._monitor_thread:
            self._stop_monitoring.set()
            self._monitor_thread.join()
            self._monitor_thread = None
            logger.info("Process monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                stats = self.get_stats()
                self._check_thresholds(stats)
                with self._lock:
                    self._stats_history.append(stats)
                    # Keep last hour of stats
                    cutoff = datetime.utcnow() - timedelta(hours=1)
                    self._stats_history = [
                        s for s in self._stats_history
                        if s.timestamp > cutoff
                    ]
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
            finally:
                self._stop_monitoring.wait(self.monitor_interval)
    
    def _check_thresholds(self, stats: SystemStats) -> None:
        """Check resource thresholds."""
        if stats.cpu_percent > self.max_cpu_percent:
            error_tracker.record_error(
                ProcessError(
                    "CPU usage above threshold",
                    resource_type="cpu",
                    current_value=stats.cpu_percent,
                    limit=self.max_cpu_percent
                )
            )
    
    def get_stats(self) -> SystemStats:
        """Get current system statistics."""
        return SystemStats(
            cpu_percent=self.process.cpu_percent(),
            memory_percent=self.process.memory_percent(),
            disk_usage={
                path: psutil.disk_usage(path).percent
                for path in self._get_monitored_paths()
            },
            network_io=dict(psutil.net_io_counters()._asdict()),
            open_files=len(self.process.open_files()),
            thread_count=self.process.num_threads(),
            process_count=len(psutil.Process().children())
        )
    
    def _get_monitored_paths(self) -> Set[str]:
        """Get paths to monitor for disk usage."""
        paths = {os.getcwd()}
        if hasattr(CONFIG.system, 'base_dir'):
            paths.add(str(CONFIG.system.base_dir))
        if hasattr(CONFIG.system, 'temp_dir'):
            paths.add(str(CONFIG.system.temp_dir))
        return paths
    
    def get_thread_stats(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get detailed thread statistics."""
        thread_stats = []
        for thread in threading.enumerate():
            thread_stats.append({
                'name': thread.name,
                'alive': thread.is_alive(),
                'daemon': thread.daemon,
                'ident': thread.ident
            })
        return {'threads': thread_stats}

class ThreadPoolManager:
    """Thread pool management with monitoring."""
    
    def __init__(
        self,
        max_workers: int = CONFIG.resources.max_threads,
        thread_timeout: int = CONFIG.resources.thread_timeout
    ):
        self.max_workers = max_workers
        self.thread_timeout = thread_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_tasks: Dict[int, datetime] = {}
        self._lock = threading.Lock()
    
    def submit(self, fn, *args, **kwargs):
        """Submit task to thread pool with monitoring."""
        future = self._executor.submit(fn, *args, **kwargs)
        
        with self._lock:
            self._active_tasks[id(future)] = datetime.utcnow()
            
            # Check for stuck tasks
            self._check_stuck_tasks()
        
        future.add_done_callback(self._task_done)
        return future
    
    def _task_done(self, future):
        """Handle completed task."""
        with self._lock:
            task_id = id(future)
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
    
    def _check_stuck_tasks(self) -> None:
        """Check for and handle stuck tasks."""
        now = datetime.utcnow()
        stuck_tasks = [
            task_id for task_id, start_time in self._active_tasks.items()
            if (now - start_time).total_seconds() > self.thread_timeout
        ]
        
        if stuck_tasks:
            error_tracker.record_error(
                ThreadError(
                    f"Found {len(stuck_tasks)} stuck tasks",
                    thread_count=len(stuck_tasks)
                )
            )
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool."""
        self._executor.shutdown(wait=wait)

# Create global managers
memory_manager = MemoryManager()
process_manager = ProcessManager()
thread_pool = ThreadPoolManager()

# Start process monitoring
process_manager.start_monitoring()

# Cleanup handler
def cleanup():
    """Cleanup system resources."""
    process_manager.stop_monitoring()
    thread_pool.shutdown()

# Register cleanup handler
import atexit
atexit.register(cleanup)

# Last modified: 2025-02-03 18:15:03 UTC
# End of system_utils.py