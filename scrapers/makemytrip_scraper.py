#!/usr/bin/env python3
"""
TourPulseAI Hotel Data Scraper
Multi-Modal Prediction System for Booking Behavior Analysis

Enhanced hotel data scraping with spatiotemporal, sentiment, and behavioral data fusion
for predictive analytics in tourism booking patterns.
"""

import csv
import json
import logging
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, parse_qs
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


@dataclass
class ReviewData:
    """Data structure for individual reviews"""
    review_id: str
    reviewer_name: str
    review_text: str
    review_rating: str
    review_date: str
    review_helpful_count: str
    hotel_id: str
    review_sentiment_score: Optional[float] = None


@dataclass
class HotelData:
    """Data structure for hotel information with multi-modal features"""
    hotel_name: str
    hotel_id: str
    scrape_timestamp: str
    location: str
    landmark: str
    distance_to_landmark: str
    user_rating: str
    rating_description: str
    review_count: str
    star_rating: str
    recent_reviews: List[str]
    review_highlights: List[str]
    price: str
    tax: str
    currency: str
    availability_status: str
    search_date: str
    checkin_date: str
    checkout_date: str
    days_ahead: int
    day_of_week: str
    season: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    price_numeric: Optional[float] = None
    review_count_numeric: Optional[int] = None
    rating_numeric: Optional[float] = None
    star_rating_numeric: Optional[int] = None
    avg_sentiment_score: Optional[float] = None


class TourPulseAIScraper:
    """Enhanced hotel scraper for TourPulseAI prediction system"""
    
    def __init__(self, headless: bool = False, log_level: str = "DEBUG", scrape_reviews: bool = True):
        self.setup_logging(log_level)
        self.driver = self._setup_driver(headless)
        self.wait = WebDriverWait(self.driver, 20)  # Increased timeout
        self.scrape_reviews = scrape_reviews
        self.all_reviews = []
        
    def setup_logging(self, level: str):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tourpulseai_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        """Setup Chrome driver with optimal configuration"""
        chrome_options = Options()
        options = [
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "--window-size=1920,1080"
        ]
        if headless:
            options.append("--headless=new")
        for option in options:
            chrome_options.add_argument(option)
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    
    def extract_temporal_features(self, url: str) -> Tuple[str, str, str, int, str, str]:
        """Extract temporal features from URL and current date"""
        try:
            parsed_url = urlparse(url)
            params = parse_qs(parsed_url.query)
            checkin = params.get('checkin', [''])[0]
            checkout = params.get('checkout', [''])[0]
            if checkin and checkout:
                checkin_date = datetime.strptime(checkin, '%m%d%Y').strftime('%Y-%m-%d')
                checkout_date = datetime.strptime(checkout, '%m%d%Y').strftime('%Y-%m-%d')
            else:
                checkin_date = checkout_date = ""
            current_date = datetime.now()
            search_date = current_date.strftime('%Y-%m-%d')
            if checkin_date:
                checkin_dt = datetime.strptime(checkin_date, '%Y-%m-%d')
                days_ahead = (checkin_dt - current_date).days
                day_of_week = checkin_dt.strftime('%A')
                month = checkin_dt.month
                if month in [12, 1, 2]:
                    season = "Winter"
                elif month in [3, 4, 5]:
                    season = "Spring"
                elif month in [6, 7, 8]:
                    season = "Summer"
                else:
                    season = "Autumn"
            else:
                days_ahead = 0
                day_of_week = ""
                season = ""
            return search_date, checkin_date, checkout_date, days_ahead, day_of_week, season
        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {e}")
            current_date = datetime.now().strftime('%Y-%m-%d')
            return current_date, "", "", 0, "", ""
    
    def extract_hotel_reviews(self, hotel_name: str, hotel_id: str, hotel_element) -> Tuple[List[str], List[str]]:
        """
        Extract at least 5 reviews and highlights from hotel listing and detailed review page
        Returns: (recent_reviews, review_highlights)
        """
        recent_reviews = []
        review_highlights = []
        min_reviews = 5
        
        try:
            # Try review snippets on listing page
            review_selectors = [
                ".userReviewCard__text",  # MakeMyTrip-specific
                ".ReviewsDisplayCard__text",
                ".ReviewCard__text",      # New potential selector
                ".review-text",
                "[data-cy='reviewText']",
                ".review__comment",
                ".review__body",
                ".user-review",
                ".comment-text",
                ".guest-review"           # Additional fallback
            ]
            for selector in review_selectors:
                try:
                    review_elements = hotel_element.find_elements(By.CSS_SELECTOR, selector)
                    self.logger.debug(f"Found {len(review_elements)} review elements with selector {selector} for {hotel_name}")
                    for elem in review_elements[:min_reviews]:
                        review_text = elem.text.strip()
                        if len(review_text) > 20 and review_text not in recent_reviews:
                            recent_reviews.append(review_text)
                    if len(recent_reviews) >= min_reviews:
                        break
                except NoSuchElementException:
                    self.logger.debug(f"No reviews found with selector {selector} on listing page for {hotel_name}")
                    continue
            
            # Extract review highlights/tags
            highlight_selectors = [
                ".tag__item",             # MakeMyTrip-specific
                ".highlightTag",
                ".reviewHighlight",
                ".review-tag",
                "[data-cy='reviewHighlight']",
                ".review__tag",
                ".amenity-tag",
                ".guest-review__tag"      # Additional fallback
            ]
            for selector in highlight_selectors:
                try:
                    highlight_elements = hotel_element.find_elements(By.CSS_SELECTOR, selector)
                    self.logger.debug(f"Found {len(highlight_elements)} highlight elements with selector {selector} for {hotel_name}")
                    for elem in highlight_elements[:5]:
                        highlight_text = elem.text.strip()
                        if highlight_text and len(highlight_text) < 50 and highlight_text not in review_highlights:
                            review_highlights.append(highlight_text)
                    if review_highlights:
                        break
                except NoSuchElementException:
                    self.logger.debug(f"No highlights found with selector {selector} for {hotel_name}")
                    continue
            
            # Scrape detailed reviews from hotel's review page
            if self.scrape_reviews and len(recent_reviews) < min_reviews:
                max_attempts = 3  # Increased retries
                for attempt in range(max_attempts):
                    try:
                        self.logger.info(f"Attempt {attempt + 1} to navigate to reviews page for {hotel_name}")
                        # Find link to hotel detail or reviews page
                        link_selectors = [
                            "a[href*='reviews']",           # Direct reviews link
                            "[data-cy='reviewsButton']",
                            "[data-testid='view-all-reviews']",
                            ".btn__review",
                            "a[href*='hotel-details']",     # Hotel detail page
                            ".hotel__link",
                            "[data-cy='hotelNameLink']",    # New MakeMyTrip-specific
                            ".listing__hotel-link"          # Additional fallback
                        ]
                        review_url = None
                        for selector in link_selectors:
                            try:
                                link_element = hotel_element.find_element(By.CSS_SELECTOR, selector)
                                if link_element.is_displayed():
                                    review_url = link_element.get_attribute('href')
                                    self.logger.debug(f"Found link: {review_url} for {hotel_name}")
                                    break
                            except NoSuchElementException:
                                self.logger.debug(f"No link found with selector {selector} for {hotel_name}")
                                continue
                        
                        if not review_url:
                            self.logger.warning(f"No reviews or detail link found for {hotel_name} after attempt {attempt + 1}")
                            continue
                        
                        # Store current window
                        main_window = self.driver.current_window_handle
                        
                        # Open page in new tab
                        self.driver.execute_script("window.open(arguments[0], '_blank');", review_url)
                        time.sleep(10)  # Increased wait for page load
                        self.driver.switch_to.window(self.driver.window_handles[-1])
                        
                        # Check for CAPTCHA
                        try:
                            page_source = self.driver.page_source.lower()
                            if any(kw in page_source for kw in ["captcha", "verify you are not a robot", "recaptcha"]):
                                self.logger.warning(f"CAPTCHA detected on reviews page for {hotel_name}. Waiting 30s for manual solve.")
                                if not self.driver.options.args.count("--headless"):
                                    time.sleep(30)  # Allow manual CAPTCHA solving in non-headless mode
                                else:
                                    self.logger.error(f"CAPTCHA detected in headless mode for {hotel_name}. Cannot proceed.")
                                    self.driver.close()
                                    self.driver.switch_to.window(main_window)
                                    return recent_reviews, review_highlights
                        except Exception as e:
                            self.logger.error(f"Error checking for CAPTCHA: {e}")
                        
                        # If on hotel detail page, navigate to reviews tab
                        if 'hotel-details' in review_url:
                            reviews_tab_selectors = [
                                "[data-cy='reviewsTab']",      # MakeMyTrip-specific
                                ".tab__reviews",
                                "a[href*='reviews']",
                                ".reviews__tab",
                                "[data-testid='reviews-tab']",
                                ".btn--reviews",
                                ".nav__reviews",               # New potential selector
                                ".hotel-reviews__tab"          # Additional fallback
                            ]
                            for tab_selector in reviews_tab_selectors:
                                try:
                                    reviews_tab = self.wait.until(
                                        EC.element_to_be_clickable((By.CSS_SELECTOR, tab_selector))
                                    )
                                    reviews_tab.click()
                                    time.sleep(5)  # Wait for reviews tab to load
                                    self.logger.debug(f"Clicked reviews tab with selector {tab_selector} for {hotel_name}")
                                    break
                                except (NoSuchElementException, TimeoutException):
                                    self.logger.debug(f"No reviews tab found with selector {tab_selector} for {hotel_name}")
                                    continue
                        
                        # Scroll multiple times to load dynamic content
                        for _ in range(5):  # Increased scrolls
                            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(3)
                        
                        # Extract detailed reviews
                        detailed_reviews = self._extract_detailed_reviews(hotel_id, hotel_name, min_reviews)
                        recent_reviews.extend([r for r in detailed_reviews if r not in recent_reviews])
                        
                        # Close tab and switch back
                        self.driver.close()
                        self.driver.switch_to.window(main_window)
                        
                        if len(recent_reviews) >= min_reviews:
                            self.logger.info(f"Successfully collected {len(recent_reviews)} reviews for {hotel_name}")
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Error navigating to reviews page for {hotel_name} on attempt {attempt + 1}: {e}")
                        if self.driver.current_window_handle != main_window:
                            self.driver.close()
                            self.driver.switch_to.window(main_window)
                        if attempt == max_attempts - 1:
                            self.logger.warning(f"Failed to get {min_reviews} reviews for {hotel_name} after {max_attempts} attempts")
                        time.sleep(3)  # Wait before retry
                        continue
        
        except Exception as e:
            self.logger.error(f"Error extracting reviews for {hotel_name}: {e}")
        
        # Ensure at least 5 valid reviews, use placeholders only as last resort
        valid_reviews = [r for r in recent_reviews if not r.startswith("No review available")]
        if len(valid_reviews) < min_reviews:
            self.logger.warning(f"Only {len(valid_reviews)} valid reviews found for {hotel_name}. Adding placeholders.")
            for i in range(len(valid_reviews), min_reviews):
                review_text = f"No review available {i + 1}"
                recent_reviews.append(review_text)
                self.all_reviews.append(ReviewData(
                    review_id=f"{hotel_id}_review_{i + 1}",
                    reviewer_name="Anonymous",
                    review_text=review_text,
                    review_rating="",
                    review_date="",
                    review_helpful_count="",
                    hotel_id=hotel_id
                ))
        
        # Calculate sentiment score (excluding placeholders)
        avg_sentiment = self.calculate_sentiment_score(valid_reviews)
        for review in self.all_reviews:
            if review.hotel_id == hotel_id:
                review.review_sentiment_score = avg_sentiment
        
        return recent_reviews[:min_reviews], review_highlights
    
    def _extract_detailed_reviews(self, hotel_id: str, hotel_name: str, min_reviews: int = 5) -> List[str]:
        """Extract at least min_reviews from reviews page"""
        reviews = []
        
        try:
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
            time.sleep(7)  # Increased wait for page load
            
            review_selectors = [
                ".userReviewCard__text",   # MakeMyTrip-specific
                ".ReviewsDisplayCard__text",
                ".ReviewCard__text",       # New potential selector
                ".review-text",
                "[data-cy='reviewText']",
                ".review__comment",
                ".review__body",
                ".user-review",
                ".comment-text",
                ".guest-review",           # Additional fallback
                ".review-content"          # Additional fallback
            ]
            
            for selector in review_selectors:
                try:
                    review_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    self.logger.debug(f"Found {len(review_elements)} detailed review elements with selector {selector} for {hotel_name}")
                    for i, elem in enumerate(review_elements[:min_reviews * 2]):  # Get extra to filter duplicates
                        review_text = elem.text.strip()
                        if len(review_text) > 20 and review_text not in reviews:
                            reviews.append(review_text)
                            review_data = ReviewData(
                                review_id=f"{hotel_id}_review_{len(reviews)}",
                                reviewer_name="Anonymous",
                                review_text=review_text,
                                review_rating="",
                                review_date="",
                                review_helpful_count="",
                                hotel_id=hotel_id
                            )
                            try:
                                parent = elem.find_element(By.XPATH, "./..")
                                rating_selectors = [
                                    ".rating",
                                    ".star-rating",
                                    "[data-cy='rating']",
                                    ".review__rating",
                                    "[data-testid='rating']",
                                    ".rating__value",
                                    ".review-score",       # New potential selector
                                    ".guest-rating"        # Additional fallback
                                ]
                                for rating_sel in rating_selectors:
                                    try:
                                        rating_elem = parent.find_element(By.CSS_SELECTOR, rating_sel)
                                        review_data.review_rating = rating_elem.text.strip()
                                        break
                                    except NoSuchElementException:
                                        continue
                                date_selectors = [
                                    ".date",
                                    ".review-date",
                                    "[data-cy='review-date']",
                                    ".review__date",
                                    "[data-testid='review-date']",
                                    ".posted__date",
                                    ".review-posted-date"  # New potential selector
                                ]
                                for date_sel in date_selectors:
                                    try:
                                        date_elem = parent.find_element(By.CSS_SELECTOR, date_sel)
                                        review_data.review_date = date_elem.text.strip()
                                        break
                                    except NoSuchElementException:
                                        continue
                                name_selectors = [
                                    ".reviewer-name",
                                    ".user-name",
                                    "[data-cy='reviewer-name']",
                                    ".review__author",
                                    "[data-testid='reviewer-name']",
                                    ".author__name",
                                    ".guest-name"          # Additional fallback
                                ]
                                for name_sel in name_selectors:
                                    try:
                                        name_elem = parent.find_element(By.CSS_SELECTOR, name_sel)
                                        review_data.reviewer_name = name_elem.text.strip()
                                        break
                                    except NoSuchElementException:
                                        continue
                                helpful_selectors = [
                                    ".helpful-count",
                                    ".vote-count",
                                    "[data-cy='helpful-count']",
                                    ".review__helpful",
                                    "[data-testid='helpful-count']",
                                    ".helpful__count",
                                    ".helpful-votes"       # New potential selector
                                ]
                                for helpful_sel in helpful_selectors:
                                    try:
                                        helpful_elem = parent.find_element(By.CSS_SELECTOR, helpful_sel)
                                        review_data.review_helpful_count = helpful_elem.text.strip()
                                        break
                                    except NoSuchElementException:
                                        continue
                            except NoSuchElementException:
                                pass
                            self.all_reviews.append(review_data)
                    if len(reviews) >= min_reviews:
                        break
                except NoSuchElementException:
                    self.logger.debug(f"No detailed reviews found with selector {selector} for {hotel_name}")
                    continue
            
            # Click "Show More" up to 5 times to load more reviews
            for attempt in range(5):
                if len(reviews) >= min_reviews:
                    break
                try:
                    show_more_selectors = [
                        ".show-more-reviews",
                        "[data-cy='showMoreReviews']",
                        ".load-more",
                        ".btn--load-more",
                        "[data-testid='load-more-reviews']",
                        ".more-reviews",
                        ".showMoreReviews",        # New potential selector
                        ".loadMoreReviews"         # Additional fallback
                    ]
                    for selector in show_more_selectors:
                        try:
                            show_more_button = self.wait.until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                            )
                            show_more_button.click()
                            time.sleep(5)  # Increased wait for reviews to load
                            self.logger.debug(f"Clicked 'Show More' button with selector {selector} (attempt {attempt + 1}) for {hotel_name}")
                            break
                        except (NoSuchElementException, TimeoutException):
                            continue
                    # Re-run review extraction
                    for selector in review_selectors:
                        try:
                            review_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            self.logger.debug(f"Found {len(review_elements)} additional review elements after 'Show More' (attempt {attempt + 1}) for {hotel_name}")
                            for i, elem in enumerate(review_elements[len(reviews):min_reviews * 2]):
                                review_text = elem.text.strip()
                                if len(review_text) > 20 and review_text not in reviews:
                                    reviews.append(review_text)
                                    review_data = ReviewData(
                                        review_id=f"{hotel_id}_review_{len(reviews)}",
                                        reviewer_name="Anonymous",
                                        review_text=review_text,
                                        review_rating="",
                                        review_date="",
                                        review_helpful_count="",
                                        hotel_id=hotel_id
                                    )
                                    self.all_reviews.append(review_data)
                            if len(reviews) >= min_reviews:
                                break
                        except NoSuchElementException:
                            continue
                except (NoSuchElementException, TimeoutException):
                    self.logger.debug(f"No 'Show More' button found (attempt {attempt + 1}) for {hotel_name}")
                    break
        
        except Exception as e:
            self.logger.error(f"Error extracting detailed reviews for {hotel_name}: {e}")
        
        return reviews[:min_reviews]
    
    def calculate_sentiment_score(self, reviews: List[str]) -> float:
        """Basic sentiment analysis using keyword-based approach"""
        if not reviews:
            return 0.0
        positive_words = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'nice',
            'clean', 'comfortable', 'friendly', 'helpful', 'recommend', 'perfect',
            'beautiful', 'spacious', 'convenient', 'professional', 'outstanding',
            'love', 'loved', 'enjoy', 'enjoyed', 'pleasant', 'satisfied'
        ]
        negative_words = [
            'terrible', 'awful', 'bad', 'worst', 'horrible', 'dirty', 'uncomfortable',
            'rude', 'unhelpful', 'disappointing', 'poor', 'cheap', 'noisy', 'smelly',
            'broken', 'outdated', 'cramped', 'overpriced', 'hate', 'hated', 'annoying',
            'disgusting', 'unacceptable', 'nightmare', 'avoid'
        ]
        total_score = 0
        word_count = 0
        for review in reviews:
            review_lower = review.lower()
            words = review_lower.split()
            for word in words:
                word_count += 1
                if word in positive_words:
                    total_score += 1
                elif word in negative_words:
                    total_score -= 1
        if word_count == 0:
            return 0.0
        return max(-1.0, min(1.0, total_score / word_count * 10))
    
    def extract_numeric_features(self, hotel_data: HotelData) -> HotelData:
        """Extract numeric features for ML model"""
        try:
            price_match = re.findall(r'[\d,]+', hotel_data.price.replace(',', ''))
            hotel_data.price_numeric = float(price_match[0]) if price_match else None
            review_match = re.findall(r'\d+', hotel_data.review_count.replace(',', ''))
            hotel_data.review_count_numeric = int(review_match[0]) if review_match else None
            rating_match = re.findall(r'\d+\.?\d*', hotel_data.user_rating)
            hotel_data.rating_numeric = float(rating_match[0]) if rating_match else None
            if hotel_data.star_rating:
                hotel_data.star_rating_numeric = len(hotel_data.star_rating)
            hotel_data.avg_sentiment_score = self.calculate_sentiment_score(
                [r for r in hotel_data.recent_reviews if not r.startswith("No review available")]
            )
        except Exception as e:
            self.logger.error(f"Error extracting numeric features: {e}")
        return hotel_data
    
    def scrape_hotel_data(self, mmt_url: str, max_hotels: int = 50, delay: float = 3.0) -> List[HotelData]:
        """
        Scrape hotel data with enhanced features for TourPulseAI
        """
        hotels_data = []
        try:
            self.logger.info(f"Starting scrape for URL: {mmt_url}")
            self.driver.get(mmt_url)
            time.sleep(10)  # Increased initial page load
            search_date, checkin_date, checkout_date, days_ahead, day_of_week, season = self.extract_temporal_features(mmt_url)
            for i in range(max_hotels):
                try:
                    self.logger.info(f"Scraping hotel {i+1}/{max_hotels}")
                    hotel_xpath = f'//*[@id="Listing_hotel_{i}"]'
                    try:
                        hotel_element = self.wait.until(
                            EC.presence_of_element_located((By.XPATH, hotel_xpath))
                        )
                    except TimeoutException:
                        self.logger.warning(f"Hotel {i} not found, stopping scrape")
                        break
                    hotel_name = self._safe_extract(hotel_element, By.ID, 'hlistpg_hotel_name')
                    hotel_id = f"hotel_{i}_{int(time.time())}"
                    user_rating = self._safe_extract(hotel_element, By.ID, 'hlistpg_hotel_user_rating')
                    review_count = self._safe_extract(hotel_element, By.ID, 'hlistpg_hotel_reviews_count')
                    rating_desc = ""
                    rating_desc_xpaths = [
                        f'//*[@id="Listing_hotel_{i}"]/a/div/div[1]/div[2]/div[1]/div/div/span[1]',
                        f'//*[@id="Listing_hotel_{i}"]/a/div/div/div[1]/div[2]/div[2]/div/div/span[2]'
                    ]
                    for xpath in rating_desc_xpaths:
                        try:
                            rating_desc_element = hotel_element.find_element(By.XPATH, xpath)
                            rating_desc = rating_desc_element.text
                            break
                        except NoSuchElementException:
                            continue
                    location_info = self._extract_location_info(hotel_element)
                    price = self._safe_extract(hotel_element, By.ID, 'hlistpg_hotel_shown_price')
                    if price.startswith('₹'):
                        price = price[1:]
                        currency = "INR"
                    else:
                        currency = ""
                    tax = ""
                    try:
                        tax_xpath = f'//*[@id="Listing_hotel_{i}"]/a/div[1]/div/div[2]/div/div/p[2]'
                        tax_element = hotel_element.find_element(By.XPATH, tax_xpath)
                        tax_parts = tax_element.text.split(" ")
                        tax = tax_parts[2] if len(tax_parts) > 2 else ""
                    except NoSuchElementException:
                        pass
                    star_rating = ""
                    try:
                        star_element = hotel_element.find_element(By.ID, 'hlistpg_hotel_star_rating')
                        star_rating = star_element.get_attribute('data-content') or ""
                    except NoSuchElementException:
                        pass
                    recent_reviews, review_highlights = self.extract_hotel_reviews(
                        hotel_name, hotel_id, hotel_element
                    )
                    hotel_data = HotelData(
                        hotel_name=hotel_name,
                        hotel_id=hotel_id,
                        scrape_timestamp=datetime.now().isoformat(),
                        location=location_info['location'],
                        landmark=location_info['landmark'],
                        distance_to_landmark=location_info['distance'],
                        user_rating=user_rating,
                        rating_description=rating_desc,
                        review_count=review_count,
                        star_rating=star_rating,
                        recent_reviews=recent_reviews,
                        review_highlights=review_highlights,
                        price=price,
                        tax=tax,
                        currency=currency,
                        availability_status="Available",
                        search_date=search_date,
                        checkin_date=checkin_date,
                        checkout_date=checkout_date,
                        days_ahead=days_ahead,
                        day_of_week=day_of_week,
                        season=season
                    )
                    hotel_data = self.extract_numeric_features(hotel_data)
                    hotels_data.append(hotel_data)
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(delay)
                except Exception as e:
                    self.logger.error(f"Error scraping hotel {i}: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
        self.logger.info(f"Scraping completed. Total hotels: {len(hotels_data)}")
        return hotels_data
    
    def _safe_extract(self, element, by, value) -> str:
        """Safely extract text from element"""
        try:
            return element.find_element(by, value).text.strip()
        except NoSuchElementException:
            return ""
    
    def _extract_location_info(self, hotel_element) -> Dict[str, str]:
        """Extract location information with better parsing"""
        try:
            location_element = hotel_element.find_element(By.CLASS_NAME, 'pc__html')
            location_text = location_element.text.strip()
            if "|" in location_text:
                parts = location_text.split("|")
                location = parts[0].strip()
                if len(parts) > 1 and "from" in parts[1]:
                    landmark_parts = parts[1].split("from")
                    distance = landmark_parts[0].strip()
                    landmark = landmark_parts[1].strip() if len(landmark_parts) > 1 else ""
                else:
                    distance = ""
                    landmark = parts[1].strip() if len(parts) > 1 else ""
            else:
                location = location_text
                landmark = ""
                distance = ""
            return {'location': location, 'landmark': landmark, 'distance': distance}
        except NoSuchElementException:
            return {'location': '', 'landmark': '', 'distance': ''}
    
    def save_data(self, hotels_data: List[HotelData], output_format: str = "both", 
                  csv_path: str = "tourpulseai_hotels.csv", 
                  json_path: str = "tourpulseai_hotels.json",
                  reviews_csv_path: str = "tourpulseai_reviews.csv"):
        """
        Save scraped data in multiple formats
        """
        try:
            if not hotels_data:
                self.logger.warning("No data to save")
                return
            if output_format in ["csv", "both"]:
                self._save_csv(hotels_data, csv_path)
                if self.all_reviews:
                    self._save_reviews_csv(reviews_csv_path)
            if output_format in ["json", "both"]:
                self._save_json(hotels_data, json_path)
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def _save_reviews_csv(self, reviews_csv_path: str):
        """Save detailed reviews to separate CSV"""
        try:
            if not self.all_reviews:
                return
            df = pd.DataFrame([asdict(review) for review in self.all_reviews])
            df.to_csv(reviews_csv_path, index=False)
            self.logger.info(f"Reviews data saved to CSV: {reviews_csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving reviews CSV: {e}")
    
    def _save_csv(self, hotels_data: List[HotelData], csv_path: str):
        """Save data to CSV format"""
        try:
            df = pd.DataFrame([asdict(hotel) for hotel in hotels_data])
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Data saved to CSV: {csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
    
    def _save_json(self, hotels_data: List[HotelData], json_path: str):
        """Save data to JSON format"""
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(hotel) for hotel in hotels_data], f, indent=2, ensure_ascii=False)
            self.logger.info(f"Data saved to JSON: {json_path}")
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")
    
    def close(self):
        """Close the driver"""
        if self.driver:
            self.driver.quit()


def main():
    """Main execution function with example usage"""
    MMT_URL = "https://www.makemytrip.com/hotels/hotel-listing/?checkin=08022025&city=CTCCU&checkout=08032025&roomStayQualifier=2e0e&locusId=CTCCU&country=IN&locusType=city&searchText=Kolkata"
    scraper = TourPulseAIScraper(headless=False, log_level="DEBUG", scrape_reviews=True)
    try:
        hotels_data = scraper.scrape_hotel_data(
            mmt_url=MMT_URL,
            max_hotels=20,
            delay=3.0
        )
        scraper.save_data(
            hotels_data=hotels_data,
            output_format="both",
            csv_path="tourpulseai6_hotels.csv",
            json_path="tourpulseai6_hotels.json",
            reviews_csv_path="tourpulseai_reviews.csv"
        )
        print(f"\n=== TourPulseAI Scraping Summary ===")
        print(f"Total hotels scraped: {len(hotels_data)}")
        print(f"Total detailed reviews: {len(scraper.all_reviews)}")
        print(f"Data saved to: tourpulseai2_hotels.csv and tourpulseai2_hotels.json")
        print(f"Reviews saved to: tourpulseai2_reviews.csv")
        for hotel in hotels_data:
            print(f"\nHotel: {hotel.hotel_name}")
            print(f"Rating: {hotel.user_rating} ({hotel.rating_description})")
            print(f"Price: {hotel.currency} {hotel.price}")
            print(f"Location: {hotel.location}")
            print(f"Recent Reviews: {len(hotel.recent_reviews)} found")
            print(f"Review Highlights: {hotel.review_highlights}")
            print(f"Sentiment Score: {hotel.avg_sentiment_score}")
            print("\nReviews:")
            for i, review in enumerate(hotel.recent_reviews, 1):
                print(f"{i}. {review[:100]}...")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()