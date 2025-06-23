import re
from bs4 import BeautifulSoup, Comment
from typing import Optional, List
import html
import logging

logger = logging.getLogger(__name__)

class HTMLCleaner:
    """
    Advanced HTML cleaner that removes HTML tags while preserving meaningful text structure.
    """
    
    def __init__(self):
        # HTML entities that should be converted to readable text
        self.html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
        }
        
        # Block-level elements that should create line breaks
        self.block_elements = {
            'div', 'p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'blockquote', 'section', 'article',
            'header', 'footer', 'nav', 'main', 'aside', 'figure',
            'table', 'thead', 'tbody', 'tr', 'td', 'th'
        }
        
        # Elements that should be completely removed (including content)
        self.remove_elements = {
            'script', 'style', 'meta', 'link', 'title', 'head',
            'noscript', 'iframe', 'object', 'embed'
        }
        
        # Elements that should preserve some structure
        self.list_elements = {'ul', 'ol', 'li'}
        self.header_elements = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content and return plain text with preserved structure.
        
        Args:
            html_content: Raw HTML content to clean
            
        Returns:
            Cleaned plain text content
        """
        if not html_content or not isinstance(html_content, str):
            return ""
        
        try:
            # First, handle HTML entities
            cleaned_content = self._decode_html_entities(html_content)
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(cleaned_content, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Remove comments
            self._remove_comments(soup)
            
            # Clean attributes (remove style, class, id, etc.)
            self._clean_attributes(soup)
            
            # Convert HTML structure to plain text with preserved formatting
            cleaned_text = self._convert_to_text(soup)
            
            # Post-process the text
            cleaned_text = self._post_process_text(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML content: {str(e)}")
            # Fallback: simple regex-based cleaning
            return self._fallback_html_clean(html_content)

    def _decode_html_entities(self, content: str) -> str:
        """Decode HTML entities to readable text."""
        # Use Python's html.unescape for standard entities
        content = html.unescape(content)
        
        # Handle additional custom entities
        for entity, replacement in self.html_entities.items():
            content = content.replace(entity, replacement)
        
        return content

    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from the soup."""
        for element_name in self.remove_elements:
            elements = soup.find_all(element_name)
            for element in elements:
                element.decompose()

    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """Remove HTML comments."""
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

    def _clean_attributes(self, soup: BeautifulSoup) -> None:
        """Remove unnecessary attributes from HTML elements."""
        # Attributes to keep (mostly for links and images)
        keep_attributes = {'href', 'src', 'alt', 'title'}
        
        for element in soup.find_all():
            # Get all attributes to remove
            attrs_to_remove = []
            for attr in element.attrs:
                if attr not in keep_attributes:
                    attrs_to_remove.append(attr)
            
            # Remove the attributes
            for attr in attrs_to_remove:
                del element.attrs[attr]

    def _convert_to_text(self, soup: BeautifulSoup) -> str:
        """Convert HTML soup to structured plain text."""
        result_parts = []
        
        def extract_text_recursive(element, depth=0):
            """Recursively extract text from elements."""
            if hasattr(element, 'name') and element.name:
                tag_name = element.name.lower()
                
                # Handle different types of elements
                if tag_name in self.header_elements:
                    # Headers get double newlines and the text
                    text_content = element.get_text().strip()
                    if text_content:
                        result_parts.append(f'\n\n{text_content}\n')
                
                elif tag_name == 'p':
                    # Paragraphs get their text with newlines
                    text_content = element.get_text().strip()
                    if text_content:
                        result_parts.append(f'\n{text_content}\n')
                
                elif tag_name == 'li':
                    # List items get bullet points
                    text_content = element.get_text().strip()
                    if text_content:
                        result_parts.append(f'\n• {text_content}')
                
                elif tag_name in {'ul', 'ol'}:
                    # Process list children
                    result_parts.append('\n')
                    for child in element.children:
                        if hasattr(child, 'name'):
                            extract_text_recursive(child, depth + 1)
                    result_parts.append('\n')
                
                elif tag_name == 'a' and element.get('href'):
                    # Links - preserve both text and URL
                    link_text = element.get_text().strip()
                    href = element.get('href')
                    if link_text:
                        if href and href != link_text and not href.startswith('#'):
                            result_parts.append(f'{link_text} ({href})')
                        else:
                            result_parts.append(link_text)
                
                elif tag_name in self.block_elements:
                    # Other block elements - extract text with spacing
                    text_content = element.get_text().strip()
                    if text_content and tag_name not in {'ul', 'ol', 'li'}:  # Avoid double processing
                        result_parts.append(f'\n{text_content}\n')
                
                else:
                    # Inline elements - just extract text
                    for child in element.children:
                        if hasattr(child, 'name'):
                            extract_text_recursive(child, depth + 1)
                        else:
                            # Direct text content
                            text = str(child).strip()
                            if text:
                                result_parts.append(text)
            
            else:
                # It's direct text content
                text = str(element).strip()
                if text and depth == 0:  # Only add top-level text
                    result_parts.append(text)
        
        # Process the soup
        for child in soup.children:
            extract_text_recursive(child)
        
        # Join and clean up
        result = ''.join(result_parts)
        return result

    def _post_process_text(self, text: str) -> str:
        """Post-process the cleaned text to improve readability."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Replace multiple line breaks with double
        text = re.sub(r'^\s+|\s+$', '', text)  # Strip leading/trailing whitespace
        
        # Clean up bullet points
        text = re.sub(r'\n\s*•\s*\n', '\n• ', text)
        text = re.sub(r'\n\s*•\s+', '\n• ', text)
        
        # Remove standalone punctuation lines
        text = re.sub(r'\n[^\w\s]*\n', '\n', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()

    def _fallback_html_clean(self, html_content: str) -> str:
        """Fallback HTML cleaning using regex when BeautifulSoup fails."""
        try:
            # Remove script and style tags with their content
            html_content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML comments
            html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
            
            # Replace common block elements with line breaks
            block_pattern = r'</?(?:div|p|br|h[1-6]|ul|ol|li|table|tr|td|th)[^>]*>'
            html_content = re.sub(block_pattern, '\n', html_content, flags=re.IGNORECASE)
            
            # Remove all remaining HTML tags
            html_content = re.sub(r'<[^>]+>', '', html_content)
            
            # Decode HTML entities
            html_content = html.unescape(html_content)
            
            # Clean up whitespace
            html_content = re.sub(r'\s+', ' ', html_content)
            html_content = re.sub(r'\n\s*\n+', '\n\n', html_content)
            
            return html_content.strip()
            
        except Exception as e:
            logger.error(f"Error in fallback HTML cleaning: {str(e)}")
            return html_content

    def is_html_content(self, content: str) -> bool:
        """
        Check if content contains HTML tags.
        
        Args:
            content: Text content to check
            
        Returns:
            True if content contains HTML tags, False otherwise
        """
        if not content:
            return False
        
        # Look for HTML tags
        html_pattern = r'<[^>]+>'
        return bool(re.search(html_pattern, content))

    def extract_links(self, html_content: str) -> List[dict]:
        """
        Extract all links from HTML content.
        
        Args:
            html_content: HTML content to extract links from
            
        Returns:
            List of dictionaries containing link information
        """
        links = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for link in soup.find_all('a', href=True):
                link_data = {
                    'text': link.get_text().strip(),
                    'url': link['href'],
                    'title': link.get('title', '').strip()
                }
                if link_data['text'] or link_data['url']:
                    links.append(link_data)
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
        
        return links

    def clean_and_extract_metadata(self, html_content: str) -> dict:
        """
        Clean HTML content and extract useful metadata.
        
        Args:
            html_content: HTML content to process
            
        Returns:
            Dictionary containing cleaned text and metadata
        """
        result = {
            'cleaned_text': '',
            'has_html': False,
            'links': [],
            'images': [],
            'text_length': 0,
            'word_count': 0
        }
        
        if not html_content:
            return result
        
        result['has_html'] = self.is_html_content(html_content)
        
        if result['has_html']:
            # Clean the HTML
            result['cleaned_text'] = self.clean_html(html_content)
            
            # Extract links
            result['links'] = self.extract_links(html_content)
            
            # Extract images
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                for img in soup.find_all('img'):
                    img_data = {
                        'src': img.get('src', ''),
                        'alt': img.get('alt', ''),
                        'title': img.get('title', '')
                    }
                    if img_data['src']:
                        result['images'].append(img_data)
            except Exception as e:
                logger.error(f"Error extracting images: {str(e)}")
        else:
            result['cleaned_text'] = html_content
        
        # Calculate text statistics
        result['text_length'] = len(result['cleaned_text'])
        result['word_count'] = len(result['cleaned_text'].split()) if result['cleaned_text'] else 0
        
        return result 