"""
AsyncProcessor - Convert blocking operations to async for better performance
Removes blocking operations that freeze the entire application
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any, Optional, Dict, Coroutine
from functools import wraps, partial
import logging
import aiohttp
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AsyncProcessor:
    """Convert blocking operations to async for non-blocking execution"""
    
    def __init__(self, max_workers: int = 10, max_processes: int = 4):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        self.background_tasks = set()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def replace_blocking_sleep(self, seconds: float) -> None:
        """
        Replace blocking time.sleep with async version
        
        BEFORE:
        time.sleep(60)  # Blocks entire thread
        
        AFTER:
        await async_processor.replace_blocking_sleep(60)  # Non-blocking
        """
        await asyncio.sleep(seconds)
        logger.debug(f"Async sleep completed: {seconds} seconds")
    
    async def process_heavy_calculation(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Run CPU-intensive tasks in process pool without blocking
        """
        loop = asyncio.get_event_loop()
        
        # Use partial to prepare function with arguments
        prepared_func = partial(func, *args, **kwargs)
        
        try:
            result = await loop.run_in_executor(
                self.process_pool,
                prepared_func
            )
            logger.info(f"Heavy calculation {func.__name__} completed")
            return result
        except Exception as e:
            logger.error(f"Heavy calculation failed: {e}")
            raise
    
    async def run_io_operation(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Run I/O operations in thread pool without blocking
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.thread_pool,
                func,
                *args,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"I/O operation failed: {e}")
            raise
    
    async def batch_process_async(
        self,
        items: List[Any],
        processor: Callable,
        max_concurrent: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process items concurrently with controlled concurrency
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        total = len(items)
        completed = 0
        
        async def process_with_semaphore(item, index):
            async with semaphore:
                try:
                    # Handle both sync and async processors
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item)
                    else:
                        result = await self.run_io_operation(processor, item)
                    
                    nonlocal completed
                    completed += 1
                    
                    if progress_callback:
                        await progress_callback(completed, total)
                    
                    logger.debug(f"Processed item {index + 1}/{total}")
                    return index, result
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    return index, None
        
        # Create tasks for all items
        tasks = [
            process_with_semaphore(item, i) 
            for i, item in enumerate(items)
        ]
        
        # Process all tasks
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by original index
        sorted_results = sorted(
            [(idx, res) for idx, res in task_results if not isinstance(res, Exception)],
            key=lambda x: x[0]
        )
        
        return [res for _, res in sorted_results]
    
    async def fetch_url_async(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Non-blocking HTTP request
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                data = await response.json()
                return {
                    'status': response.status,
                    'data': data,
                    'headers': dict(response.headers)
                }
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            raise
    
    async def parallel_fetch(
        self,
        urls: List[str],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple URLs in parallel
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(url):
            async with semaphore:
                return await self.fetch_url_async(url)
        
        results = await asyncio.gather(
            *[fetch_with_semaphore(url) for url in urls],
            return_exceptions=True
        )
        
        # Filter out exceptions
        return [r for r in results if not isinstance(r, Exception)]
    
    async def read_file_async(self, filepath: str) -> str:
        """
        Non-blocking file read
        """
        loop = asyncio.get_event_loop()
        
        def read_file():
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        return await loop.run_in_executor(self.thread_pool, read_file)
    
    async def write_file_async(self, filepath: str, content: str) -> None:
        """
        Non-blocking file write
        """
        loop = asyncio.get_event_loop()
        
        def write_file():
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        await loop.run_in_executor(self.thread_pool, write_file)
    
    async def load_dataframe_async(
        self,
        filepath: str,
        file_type: str = 'csv'
    ) -> pd.DataFrame:
        """
        Non-blocking DataFrame loading
        """
        loop = asyncio.get_event_loop()
        
        def load_df():
            if file_type == 'csv':
                return pd.read_csv(filepath)
            elif file_type == 'excel':
                return pd.read_excel(filepath)
            elif file_type == 'json':
                return pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        
        return await loop.run_in_executor(self.thread_pool, load_df)
    
    async def save_dataframe_async(
        self,
        df: pd.DataFrame,
        filepath: str,
        file_type: str = 'csv'
    ) -> None:
        """
        Non-blocking DataFrame saving
        """
        loop = asyncio.get_event_loop()
        
        def save_df():
            if file_type == 'csv':
                df.to_csv(filepath, index=False)
            elif file_type == 'excel':
                df.to_excel(filepath, index=False)
            elif file_type == 'json':
                df.to_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        
        await loop.run_in_executor(self.thread_pool, save_df)
    
    def create_background_task(
        self,
        coro: Coroutine,
        name: Optional[str] = None
    ) -> asyncio.Task:
        """
        Create a background task that runs independently
        """
        task = asyncio.create_task(coro, name=name)
        
        # Track task to prevent garbage collection
        self.background_tasks.add(task)
        
        # Remove from set when complete
        task.add_done_callback(self.background_tasks.discard)
        
        logger.info(f"Created background task: {name or 'unnamed'}")
        return task
    
    async def wait_for_tasks(
        self,
        tasks: List[asyncio.Task],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Wait for multiple tasks with optional timeout
        """
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            return results
        except asyncio.TimeoutError:
            logger.warning(f"Tasks timed out after {timeout} seconds")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
    
    async def retry_async(
        self,
        func: Callable,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry async function with exponential backoff
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return await self.run_io_operation(func, *args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")
        
        raise last_exception
    
    async def cleanup(self):
        """
        Cleanup resources
        """
        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close session
        if self.session:
            await self.session.close()
        
        # Shutdown executors
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
        
        logger.info("AsyncProcessor cleanup completed")


class BackgroundScheduler:
    """Schedule and manage background tasks without blocking"""
    
    def __init__(self):
        self.tasks = {}
        self.running = False
        self.scheduler_task = None
        
    def add_periodic_task(
        self,
        name: str,
        func: Callable,
        interval: float,
        run_immediately: bool = False
    ):
        """
        Add a task to run periodically
        
        Args:
            name: Task identifier
            func: Async function to run
            interval: Seconds between runs
            run_immediately: Run once immediately
        """
        self.tasks[name] = {
            'func': func,
            'interval': interval,
            'last_run': None if not run_immediately else 0,
            'next_run': time.time() if run_immediately else time.time() + interval,
            'running': False,
            'run_count': 0,
            'errors': 0
        }
        logger.info(f"Added periodic task '{name}' with {interval}s interval")
    
    def add_cron_task(
        self,
        name: str,
        func: Callable,
        cron_expression: str
    ):
        """
        Add a task with cron-like scheduling
        """
        # This would use a cron parser in production
        # Simplified for demonstration
        self.tasks[name] = {
            'func': func,
            'cron': cron_expression,
            'type': 'cron',
            'run_count': 0,
            'errors': 0
        }
    
    async def start(self):
        """
        Start the background scheduler
        """
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("Background scheduler started")
    
    async def stop(self):
        """
        Stop the background scheduler
        """
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background scheduler stopped")
    
    async def _run_scheduler(self):
        """
        Main scheduler loop
        """
        while self.running:
            current_time = time.time()
            
            # Check each task
            for name, task in self.tasks.items():
                if task.get('type') == 'cron':
                    continue  # Handle cron tasks differently
                
                # Skip if task is already running
                if task['running']:
                    continue
                
                # Check if it's time to run
                if current_time >= task['next_run']:
                    # Run task in background
                    asyncio.create_task(self._run_task(name, task))
                    
                    # Update next run time
                    task['next_run'] = current_time + task['interval']
                    task['last_run'] = current_time
            
            # Check every second
            await asyncio.sleep(1)
    
    async def _run_task(self, name: str, task: Dict):
        """
        Run individual task with error handling
        """
        task['running'] = True
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(task['func']):
                await task['func']()
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, task['func'])
            
            task['run_count'] += 1
            elapsed = time.time() - start_time
            logger.info(f"Task '{name}' completed in {elapsed:.2f}s (run #{task['run_count']})")
            
        except Exception as e:
            task['errors'] += 1
            logger.error(f"Task '{name}' failed: {e}")
        
        finally:
            task['running'] = False
    
    def get_task_status(self, name: str) -> Optional[Dict]:
        """
        Get status of a specific task
        """
        if name in self.tasks:
            task = self.tasks[name]
            return {
                'name': name,
                'interval': task.get('interval'),
                'last_run': task.get('last_run'),
                'next_run': task.get('next_run'),
                'running': task.get('running', False),
                'run_count': task.get('run_count', 0),
                'errors': task.get('errors', 0)
            }
        return None
    
    def remove_task(self, name: str):
        """
        Remove a scheduled task
        """
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Removed task '{name}'")


def make_async(func: Callable) -> Callable:
    """
    Decorator to convert synchronous function to async
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    return async_wrapper


def non_blocking(func: Callable) -> Callable:
    """
    Decorator to ensure function doesn't block event loop
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if function contains blocking calls
        func_code = str(func.__code__.co_code)
        
        blocking_patterns = ['time.sleep', 'requests.', 'urllib.']
        for pattern in blocking_patterns:
            if pattern in func_code:
                logger.warning(f"Function {func.__name__} may contain blocking call: {pattern}")
        
        # Run function
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run in thread pool to prevent blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    return wrapper