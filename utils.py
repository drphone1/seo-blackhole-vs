# utils.py
# Created: 2025-02-03 11:52:57
# Last modified: 2025-02-03 11:52:57
# Author: drphon
# Repository: drphon/chat-6-deepseek
# Description: Enhanced utility functions and classes for Windows web scraping with improved performance

import os
import re
import sys
import time
import psutil
import winreg
import win32api
import win32con
import win32file
import win32event
import win32process
import win32security
import win32service 
import win32pdh
import win32job
import logging
import threading
import pythoncom
import win32com.client
import asyncio
import aiohttp
import fnmatch
import aiofiles
from typing import Optional, Dict, List, Any, Union, Tuple, Generator, Callable
from pathlib import Path, WindowsPath
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import CONFIG  
from exceptions import WindowsError, ResourceError, CacheError

# Enhanced logging configuration with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.RotatingFileHandler(
            'app.log', 
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WindowsSystemMonitor:
    """Advanced Windows system monitoring and management with enhanced error handling,
    performance metrics tracking, and improved resource management.
    
    Features:
    - Real-time system metrics monitoring
    - Performance counter tracking
    - Process and thread management
    - Enhanced error handling and logging
    - Automatic resource cleanup
    """
    
    def __init__(self, metrics_retention: int = CONFIG['monitoring'].get('metrics_retention', 1000)):
        self._initialize_com()
        self.wmi = win32com.client.GetObject("winmgmts:")
        self._setup_performance_counters()
        self.metrics_history: List[Dict[str, Any]] = []
        self.metrics_retention = metrics_retention
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def _initialize_com(self) -> None:
        try:
            pythoncom.CoInitialize()
            logger.info("COM initialization successful")
        except Exception as e:
            logger.critical(f"COM initialization failed: {str(e)}", exc_info=True)
            raise WindowsError("Critical COM initialization failure", original_error=e)
            
    def _setup_performance_counters(self) -> None:
        try:
            self.query = win32pdh.OpenQuery()
            self.counters = {
                'cpu': win32pdh.AddCounter(self.query, "\\Processor(_Total)\\% Processor Time"),
                'memory': win32pdh.AddCounter(self.query, "\\Memory\\Available MBytes"),
                'disk': win32pdh.AddCounter(self.query, "\\PhysicalDisk(_Total)\\% Disk Time"),
                'network': win32pdh.AddCounter(self.query, "\\Network Interface(*)\\Bytes Total/sec"),
                'system': win32pdh.AddCounter(self.query, "\\System\\Processor Queue Length")
            }
            logger.info("Performance counters initialized successfully")
        except Exception as e:
            logger.error(f"Performance counter setup failed: {str(e)}", exc_info=True)
            raise WindowsError("Performance counter initialization failed", original_error=e)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics with enhanced error handling and validation."""
        try:
            with self._lock:
                win32pdh.CollectQueryData(self.query)
                metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'cpu_percent': self._get_cpu_metrics(),
                    'memory': self._get_memory_metrics(),
                    'disk': self._get_disk_metrics(),
                    'network': self._get_network_metrics(),
                    'handles': self._get_handle_count(),
                    'processes': len(psutil.pids()),
                    'threads': self._get_thread_count(),
                    'performance_counters': self._get_performance_counters(),
                    'system_uptime': self._get_system_uptime()
                }
                
                self._update_metrics_history(metrics)
                return metrics
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}", exc_info=True)
            raise ResourceError("System metrics collection failed", original_error=e)

    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get detailed CPU metrics with per-core information."""
        try:
            return {
                'total_percent': psutil.cpu_percent(interval=1),
                'per_cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
                'frequency': psutil.cpu_freq()._asdict() if hasattr(psutil, 'cpu_freq') else None,
                'stats': psutil.cpu_stats()._asdict()
            }
        except Exception as e:
            logger.warning(f"CPU metrics collection partial failure: {str(e)}")
            return {'total_percent': -1}

    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get comprehensive memory metrics with swap information."""
        try:
            virtual = psutil.virtual_memory()._asdict()
            swap = psutil.swap_memory()._asdict()
            return {
                'virtual': virtual,
                'swap': swap,
                'percent_used': virtual['percent'],
                'available_gb': virtual['available'] / (1024 ** 3)
            }
        except Exception as e:
            logger.warning(f"Memory metrics collection partial failure: {str(e)}")
            return {}
def _update_metrics_history(self, metrics: Dict[str, Any]) -> None:
        """Update metrics history with retention policy."""
        self.metrics_history.append(metrics)
        while len(self.metrics_history) > self.metrics_retention:
            self.metrics_history.pop(0)

    def _get_system_uptime(self) -> float:
        """Get system uptime in seconds with enhanced accuracy."""
        try:
            return time.time() - psutil.boot_time()
        except Exception as e:
            logger.warning(f"Failed to get system uptime: {str(e)}")
            return -1

class MemoryManager:
    """Enhanced memory management for Windows with proactive monitoring and optimization.
    
    Features:
    - Proactive memory pressure detection
    - Intelligent garbage collection
    - Process priority management
    - Memory leak detection
    - Resource usage optimization
    """
    
    def __init__(self, 
                 threshold: float = CONFIG['resources']['max_memory_percent'],
                 leak_detection: bool = True):
        self.threshold = threshold
        self.warning_threshold = threshold * 0.8
        self.critical_threshold = threshold * 0.95
        self.leak_detection = leak_detection
        self.process_memory_history: Dict[int, List[float]] = {}
        self._setup_memory_monitoring()
        self._last_gc_time = time.time()
        self._gc_interval = CONFIG['resources']['gc_interval']
        self._lock = threading.Lock()

    def _setup_memory_monitoring(self) -> None:
        """Initialize enhanced memory monitoring with leak detection."""
        try:
            self.memory_event = win32event.CreateEvent(None, 0, 0, None)
            self.monitor_thread = threading.Thread(
                target=self._monitor_memory,
                daemon=True,
                name="MemoryMonitor"
            )
            self.monitor_thread.start()
            logger.info("Enhanced memory monitoring initialized with leak detection")
        except Exception as e:
            logger.error(f"Memory monitoring setup failed: {str(e)}", exc_info=True)
            raise WindowsError("Memory monitoring initialization failed", original_error=e)

    def _monitor_memory(self) -> None:
        """Enhanced memory monitoring with leak detection and optimization."""
        while True:
            try:
                with self._lock:
                    memory = psutil.virtual_memory()
                    self._update_process_memory_history()
                    
                    if self.leak_detection:
                        self._check_memory_leaks()
                        
                    if memory.percent >= self.critical_threshold:
                        self._handle_critical_memory()
                    elif memory.percent >= self.threshold:
                        self._handle_memory_pressure()
                    elif memory.percent >= self.warning_threshold:
                        self._handle_memory_warning()
                    
                    self._check_periodic_gc()
                    
                time.sleep(CONFIG['monitoring']['memory_check_interval'])
            except Exception as e:
                logger.error(f"Memory monitoring error: {str(e)}", exc_info=True)
                time.sleep(60)

    def _update_process_memory_history(self) -> None:
        """Track process memory usage for leak detection."""
        try:
            for proc in psutil.process_iter(['pid', 'memory_info']):
                pid = proc.info['pid']
                mem_usage = proc.info['memory_info'].rss
                
                if pid not in self.process_memory_history:
                    self.process_memory_history[pid] = []
                    
                history = self.process_memory_history[pid]
                history.append(mem_usage)
                
                # Keep last 60 measurements
                if len(history) > 60:
                    history.pop(0)
        except Exception as e:
            logger.warning(f"Process memory history update failed: {str(e)}")

    def _check_memory_leaks(self) -> None:
        """Detect potential memory leaks in processes."""
        try:
            for pid, history in self.process_memory_history.items():
                if len(history) >= 30:  # Need at least 30 measurements
                    growth_rate = (history[-1] - history[0]) / len(history)
                    if growth_rate > CONFIG['resources']['leak_threshold']:
                        logger.warning(f"Potential memory leak detected in PID {pid}")
                        self._handle_leaking_process(pid)
        except Exception as e:
            logger.error(f"Memory leak check failed: {str(e)}")

    def _handle_leaking_process(self, pid: int) -> None:
        """Handle processes with potential memory leaks."""
        try:
            process = psutil.Process(pid)
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            logger.info(f"Reduced priority of potentially leaking process {pid}")
        except Exception as e:
            logger.error(f"Failed to handle leaking process {pid}: {str(e)}")

class FileSystemWatcher:
    """Enhanced file system monitoring with advanced event handling and filtering.
    
    Features:
    - Real-time file system monitoring
    - Pattern-based filtering
    - Event debouncing
    - Recursive directory watching
    - Enhanced error handling
    """
    
    def __init__(self, 
                 path: str, 
                 patterns: Optional[List[str]] = None,
                 recursive: bool = True):
        self.path = path
        self.patterns = patterns or ["*"]
        self.recursive = recursive
        self.observer = Observer()
        self.handler = self._create_event_handler()
        self._setup_watchdog()
        self._lock = threading.Lock()
        
    def _create_event_handler(self) -> FileSystemEventHandler:
        class CustomHandler(FileSystemEventHandler):
            def __init__(self, patterns: List[str], parent: 'FileSystemWatcher'):
                self.patterns = patterns
                self.parent = parent
                self.last_events: Dict[str, float] = {}
                
            def on_modified(self, event):
                if not event.is_directory:
                    self._handle_file_event(event, "modified")
                    
            def on_created(self, event):
                if not event.is_directory:
                    self._handle_file_event(event, "created")
                    
            def _handle_file_event(self, event, event_type: str):
                try:
                    with self.parent._lock:
                        if any(fnmatch.fnmatch(event.src_path, pattern) 
                              for pattern in self.patterns):
                            current_time = time.time()
                            last_time = self.last_events.get(event.src_path, 0)
                            
                            if current_time - last_time > CONFIG['filesystem']['event_debounce']:
                                self.last_events[event.src_path] = current_time
                                logger.info(f"File {event_type}: {event.src_path}")
                except Exception as e:
                    logger.error(f"File event handling error: {str(e)}", exc_info=True)
                    
        return CustomHandler(self.patterns, self)

class DownloadManager:
    """Enhanced asynchronous download manager with advanced features.
    
    Features:
    - Concurrent download management
    - Rate limiting
    - Automatic retries with exponential backoff
    - Cache management
    - Download progress tracking
    """
    
    def __init__(self, 
                 max_concurrent: int = CONFIG['download']['max_concurrent'],
                 rate_limit: int = CONFIG['download']['rate_limit'],
                 cache_size: int = CONFIG['download']['cache_size']):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.cache_size = cache_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None
        self.download_cache = FileCache(max_size=cache_size)
        self._lock = asyncio.Lock()
        
    async def __aenter__(self):
        """Initialize async session with optimal settings."""
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes total timeout
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of resources."""
        if self.session:
            await self.session.close()

# Last modified: 2025-02-03 11:54:30 UTC
# Modified by: drphon
# End of utils.py