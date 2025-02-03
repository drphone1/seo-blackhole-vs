# main.py
# Created: 2025-02-03 14:15:29 UTC
# Last modified: 2025-02-03 14:15:29 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import asyncio
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from web_scraper import WebScraper
from content_processor import ContentProcessor
from config import CONFIG, get_logger
from exceptions import ScrapingError, ProcessingError

logger = get_logger(__name__)

class ScrapingManager:
    """Enhanced scraping manager for coordinating web scraping operations.
    
    Features:
    - Asynchronous processing
    - Batch processing
    - Advanced error handling
    - Comprehensive reporting
    - Resource management
    - Progress tracking
    """
    
    def __init__(self, max_concurrent: int = CONFIG['CONCURRENT_TASKS']):
        self.scraper = WebScraper()
        self.processor = ContentProcessor(
            input_dir=CONFIG['OUTPUT_DIR'],
            output_dir=CONFIG['OUTPUT_DIR']
        )
        self.results: List[Dict[str, Any]] = []
        self.failed_items: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.max_concurrent = max_concurrent
        
        # Initialize statistics
        self.stats = {
            'total_keywords': 0,
            'total_urls': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'processing_time': 0
        }

    async def process_keywords(self, keywords: List[str]):
        """Process a list of keywords by searching Google and scraping results"""
        self.stats['total_keywords'] = len(keywords)
        all_urls = []
        
        # Process keywords in batches
        batch_size = CONFIG['BATCH_SIZE']
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            batch_urls = await asyncio.gather(
                *[self._process_single_keyword(keyword) for keyword in batch],
                return_exceptions=True
            )
            
            # Filter out errors and extend urls
            for urls in batch_urls:
                if isinstance(urls, list):
                    all_urls.extend(urls)
                    
            # Progress logging
            logger.info(f"Processed {i + len(batch)}/{len(keywords)} keywords")

        if all_urls:
            await self.process_url_list([item['url'] for item in all_urls])
            # Add keyword information to results
            for result in self.results:
                matching_url = next((item for item in all_urls if item['url'] == result['url']), None)
                if matching_url:
                    result['keyword'] = matching_url['keyword']

    async def _process_single_keyword(self, keyword: str) -> List[Dict[str, str]]:
        """Process a single keyword with error handling"""
        try:
            logger.info(f"Processing keyword: {keyword}")
            urls = self.scraper.search_google(keyword)
            
            if urls:
                logger.info(f"Found {len(urls)} URLs for keyword: {keyword}")
                return [{'url': url, 'keyword': keyword} for url in urls]
            else:
                logger.warning(f"No URLs found for keyword: {keyword}")
                self._add_failed_item(keyword, 'keyword', 'No URLs found')
                return []
                
        except Exception as e:
            logger.error(f"Error processing keyword {keyword}: {str(e)}")
            self._add_failed_item(keyword, 'keyword', str(e))
            return []

    async def process_url_list(self, urls: List[str]):
        """Process a list of URLs with batch processing and rate limiting"""
        self.stats['total_urls'] = len(urls)
        
        # Process URLs in batches
        batch_size = CONFIG['BATCH_SIZE']
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            
            try:
                # Process batch
                results = await self.scraper.process_urls(batch)
                if results:
                    processed_results = await self._process_results(results)
                    self.results.extend(processed_results)
                    self.stats['successful_scrapes'] += len(processed_results)
                
                # Progress logging
                logger.info(f"Processed {i + len(batch)}/{len(urls)} URLs")
                
                # Rate limiting
                await asyncio.sleep(CONFIG['RATE_LIMIT'])
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                for url in batch:
                    self._add_failed_item(url, 'url', str(e))
                self.stats['failed_scrapes'] += len(batch)

    async def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process scraped results with content processor"""
        try:
            processed_results = []
            for result in results:
                processed = await self.processor.process_content(result)
                if processed:
                    processed_results.append(processed)
            return processed_results
        except ProcessingError as e:
            logger.error(f"Processing error: {str(e)}")
            return results  # Return original results if processing fails

    def _add_failed_item(self, item: str, item_type: str, error: str):
        """Add item to failed items list with timestamp"""
        self.failed_items.append({
            'item': item,
            'type': item_type,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

    def save_final_report(self) -> Optional[Path]:
        """Save comprehensive final report with enhanced statistics"""
        try:
            timestamp = datetime.now().strftime(CONFIG['TIMESTAMP_FORMAT'])
            
            # Create report directory
            report_dir = Path(CONFIG['OUTPUT_DIR']) / 'reports' / timestamp
            report_dir.mkdir(parents=True, exist_ok=True)

            # Save successful results
            if self.results:
                # Excel format with multiple sheets
                excel_path = report_dir / 'successful_results.xlsx'
                with pd.ExcelWriter(excel_path) as writer:
                    pd.DataFrame(self.results).to_excel(writer, sheet_name='Results', index=False)
                    pd.DataFrame(self.stats, index=[0]).to_excel(writer, sheet_name='Statistics', index=False)
                
                # JSON format
                json_path = report_dir / 'successful_results.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=4)

            # Save failed items
            if self.failed_items:
                failed_path = report_dir / 'failed_items.json'
                with open(failed_path, 'w', encoding='utf-8') as f:
                    json.dump(self.failed_items, f, ensure_ascii=False, indent=4)

            # Calculate final statistics
            self.stats['processing_time'] = str(datetime.now() - self.start_time)
            
            # Generate detailed summary report
            summary = {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': self.stats['processing_time'],
                'statistics': self.stats,
                'success_rate': f"{(len(self.results) / (len(self.results) + len(self.failed_items)) * 100):.2f}%",
                'memory_usage': self._get_memory_usage(),
                'configuration': {
                    'batch_size': CONFIG['BATCH_SIZE'],
                    'rate_limit': CONFIG['RATE_LIMIT'],
                    'max_concurrent': self.max_concurrent
                }
            }

            summary_path = report_dir / 'summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)

            logger.info(f"Final report saved successfully in {report_dir}")
            logger.info(f"Success rate: {summary['success_rate']}")
            
            return report_dir

        except Exception as e:
            logger.error(f"Error saving final report: {str(e)}")
            return None

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        import psutil
        process = psutil.Process(os.getpid())
        return {
            'memory_percent': process.memory_percent(),
            'memory_info': process.memory_info()._asdict()
        }

async def main():
    # Set up argument parser with enhanced options
    parser = argparse.ArgumentParser(
        description='Advanced web scraping tool for keywords or URLs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--keywords', type=str, help='Path to JSON file containing keywords')
    group.add_argument('--urls', type=str, help='Path to JSON file containing URLs')
    parser.add_argument('--concurrent', type=int, default=CONFIG['CONCURRENT_TASKS'],
                      help='Maximum number of concurrent tasks')
    parser.add_argument('--batch-size', type=int, default=CONFIG['BATCH_SIZE'],
                      help='Batch size for processing')
    
    args = parser.parse_args()

    # Update configuration
    CONFIG['CONCURRENT_TASKS'] = args.concurrent
    CONFIG['BATCH_SIZE'] = args.batch_size

    # Initialize scraping manager
    manager = ScrapingManager(max_concurrent=args.concurrent)
    
    try:
        if args.keywords:
            try:
                with open(args.keywords, 'r', encoding='utf-8') as f:
                    keywords = json.load(f)
                    if not isinstance(keywords, list):
                        raise ValueError("Keywords file must contain a JSON array")
                    
                logger.info(f"Loaded {len(keywords)} keywords from {args.keywords}")
                await manager.process_keywords(keywords)
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in keywords file: {args.keywords}")
                return
            except Exception as e:
                logger.error(f"Error loading keywords file: {str(e)}")
                return
                
        elif args.urls:
            try:
                with open(args.urls, 'r', encoding='utf-8') as f:
                    urls = json.load(f)
                    if not isinstance(urls, list):
                        raise ValueError("URLs file must contain a JSON array")
                    
                logger.info(f"Loaded {len(urls)} URLs from {args.urls}")
                await manager.process_url_list(urls)
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in URLs file: {args.urls}")
                return
            except Exception as e:
                logger.error(f"Error loading URLs file: {str(e)}")
                return

        # Save final report
        report_dir = manager.save_final_report()
        if report_dir:
            logger.info(f"Scraping completed. Reports saved in: {report_dir}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    
    finally:
        # Cleanup
        try:
            await manager.scraper.close()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

# Last modified: 2025-02-03 14:15:29 UTC
# End of main.py