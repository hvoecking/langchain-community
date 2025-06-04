"""Loads YouTube community posts."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from json import JSONDecoder
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_NETLOCS = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
}

# Default SOCS cookie for YouTube access
DEFAULT_SOCS_COOKIE = "CAESEwgDEgk2NDg4NTY2OTgaAnJvIAEaBgiAtae0Bg"

Post = dict[str, Any]


def _extract_json_from_html(raw: str, key: str) -> dict[str, Any]:
    """
    Extract JSON object from HTML string.

    Args:
    ----
        raw: HTML string
        key: Key to extract

    Returns:
    -------
        JSON object if found, otherwise raise ValueError.

    """
    encoded_key = f'"{key}":'

    key_pos = raw.find(encoded_key)
    if key_pos == -1:
        raise ValueError(f"Key {key} not found in {raw[:100]}...")

    brace_pos = raw.find("{", key_pos + len(encoded_key))
    if brace_pos == -1:
        raise ValueError(
            f"Opening brace not found in {raw[key_pos : key_pos + 100]}..."
        )

    decoder = JSONDecoder()
    obj, _ = decoder.raw_decode(raw[brace_pos:])
    return obj


def _parse_community_post_url(url: str) -> str:
    """
    Parse post ID from YouTube community post URL if valid, otherwise raise ValueError.

    Args:
    ----
        url: YouTube community post URL

    Returns:
    -------
        Post ID if valid, otherwise raise ValueError.

    """
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}")

    if parsed_url.netloc not in ALLOWED_NETLOCS:
        raise ValueError(f"Invalid URL netloc: {parsed_url.netloc}")

    path = parsed_url.path

    # Check if this is a community post URL
    if "/post/" not in path:
        raise ValueError(f"Invalid URL path: {path}")

    return path.split("/")[-1]


def _get_socs_cookie() -> str:
    """
    Get SOCS cookie for YouTube access. Uses default if unable to obtain dynamically.

    Returns
    -------
        SOCS cookie if available, otherwise default.

    """
    try:
        session = requests.Session()

        if "SOCS" in session.cookies:
            cookie = session.cookies["SOCS"]
            if cookie and len(cookie) > 14:
                return cookie
    except Exception as e:
        logger.debug(f"Failed to get SOCS cookie: {e}, using default")

    return DEFAULT_SOCS_COOKIE


def _get_video_link(post: Post) -> str | None:
    """
    Extract video link from post data.

    Args:
    ----
        post: Post data

    Returns:
    -------
        Video link if available, otherwise None.

    """
    try:
        video_id = post["backstageAttachment"]["videoRenderer"]["videoId"]
        return f"https://www.youtube.com/watch?v={video_id}"
    except KeyError:
        return None


def _get_image_links(post: Post) -> list[str] | None:
    """
    Extract image links from post data.

    Args:
    ----
        post: Post data

    Returns:
    -------
        List of image links if available, otherwise None.

    """
    try:
        # Try single image first
        single_image = post["backstageAttachment"]["backstageImageRenderer"]["image"][
            "thumbnails"
        ][-1]["url"]
        return [single_image]
    except KeyError:
        pass

    try:
        # Try multiple images
        images = post["backstageAttachment"]["postMultiImageRenderer"]["images"]
        links = []
        for image in images:
            url = image["backstageImageRenderer"]["image"]["thumbnails"][-1]["url"]
            links.append(url)
        return links if links else None
    except KeyError:
        return None


def _handle_text_content(content: dict[str, Any]) -> str:
    """
    Handle individual text content with potential links.

    Args:
    ----
        content: Text content

    Returns:
    -------
        Text content with potential links.

    """
    try:
        # Handle navigation/link content
        link_redirect = content["navigationEndpoint"]["urlEndpoint"]["url"]
        # Extract actual URL from redirect
        link_match = re.findall(r"(?<=q=)(.+)", link_redirect)
        if link_match:
            return unquote(link_match[0])
        return link_redirect
    except (KeyError, IndexError):
        # Return plain text
        return content.get("text", "")


def _get_text_content(post: Post) -> str | None:
    """
    Extract text content from post data.

    Args:
    ----
        post: Post data

    Returns:
    -------
        Text content if available, otherwise None.

    """
    try:
        # Try main content text
        text_runs = post["contentText"]["runs"]
        strings = [_handle_text_content(content) for content in text_runs]
        return "".join(strings)
    except KeyError:
        pass

    try:
        # Try shared post content
        text_runs = post["content"]["runs"]
        strings = [_handle_text_content(content) for content in text_runs]
        return "".join(strings)
    except KeyError:
        return None


def _get_poll_content(post: Post) -> str | None:
    """
    Extract poll content from post data.

    Args:
    ----
        post: Post data

    Returns:
    -------
        Poll content if available, otherwise None.

    """
    try:
        poll = post["backstageAttachment"]["pollRenderer"]
        choices = poll.get("choices", [])
        if not choices:
            return None

        poll_text = "Poll: "
        for choice in choices:
            text = choice.get("text", {}).get("runs", [{}])[0].get("text", "")
            # Try to get vote percentage if available
            vote_ratio = choice.get("voteRatio", 0)
            if vote_ratio > 0:
                percentage = round(vote_ratio * 100, 1)
                poll_text += f"{text}: {percentage}%, "
            else:
                poll_text += f"{text}, "

        return poll_text.rstrip(", ")
    except KeyError:
        return None


class YoutubeCommunityLoader(BaseLoader):
    """
    Load `YouTube` community posts.

    This implementation uses YouTube's internal API to fetch community posts,
    similar to the yp-dl project approach.
    """

    def __init__(
        self,
        post_id: str,
        continue_on_failure: bool = False,
    ):
        """
        Initialize with YouTube community post ID.

        Args:
            post_id: YouTube community post ID
            continue_on_failure: Whether to continue on failure

        """
        self.post_id = post_id
        self.continue_on_failure = continue_on_failure

    @staticmethod
    def extract_post_id(youtube_community_url: str) -> str:
        """
        Extract post ID from YouTube community post URL.

        Args:
        ----
            youtube_community_url: YouTube community post URL

        Returns:
        -------
            Post ID if valid, otherwise raises ValueError.

        """
        return _parse_community_post_url(youtube_community_url)

    @classmethod
    def from_youtube_community_url(
        cls, youtube_community_url: str, **kwargs: Any
    ) -> YoutubeCommunityLoader:
        """
        Given a YouTube community post URL, construct a loader.

        Args:
        ----
            youtube_community_url: YouTube community post URL
            **kwargs: Additional arguments passed to constructor

        Returns:
        -------
            Loader instance.

        """
        post_id = cls.extract_post_id(youtube_community_url)
        return cls(post_id, **kwargs)

    def load(self) -> list[Document]:
        """Load YouTube community post into `Document` objects."""
        try:
            return self._load_post()
        except Exception as e:
            if self.continue_on_failure:
                logger.error(f"Error loading community post {self.post_id}: {e}")
                return []
            else:
                raise e

    def _load_post(self) -> list[Document]:
        """Load the specific community post using YouTube's internal API."""
        # Get SOCS cookie for authentication
        socs_cookie = _get_socs_cookie()
        cookies = {"SOCS": socs_cookie}

        # Construct the post URL
        post_url = f"https://www.youtube.com/post/{self.post_id}"

        session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
                "Chrome/117.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                "image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        response = session.get(post_url, headers=headers, cookies=cookies, timeout=10)
        response.raise_for_status()

        post_data = self._extract_post_from_html(response.text, post_url)

        return [self._create_document(post_data)]

    def _extract_post_from_html(self, html: str, post_url: str) -> Post:
        """Extract post data from HTML response."""
        # Look for JSON data containing the post
        # YouTube embeds post data in script tags
        post_json = _extract_json_from_html(html, "backstagePostRenderer")

        if not post_json:
            # Try alternative pattern for shared posts
            post_json = _extract_json_from_html(html, "sharedPostRenderer")

        return self._parse_post_data(post_json, post_url)

    def _parse_post_data(self, post_json: Post, post_url: str) -> Post:
        """Parse post JSON data into structured format."""
        post_data = {
            "post_link": post_url,
            "post_id": self.post_id,
            "text_content": _get_text_content(post_json) or "",
            "video_link": _get_video_link(post_json),
            "image_links": _get_image_links(post_json),
            "poll_content": _get_poll_content(post_json),
            "time_of_download": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        post_data["time_since"] = post_json["publishedTimeText"]["runs"][0]["text"]

        return post_data

    def _create_document(self, post_data: Post) -> Document:
        """
        Create a Document from post data.

        Args:
        ----
            post_data: Post data

        Returns:
        -------
            Document object containing post data.

        """
        text_content = post_data.get("text_content", "")

        metadata = {
            "source": post_data.get("post_link", ""),
            "post_id": self.post_id,
            "time_since": post_data.get("time_since", ""),
            "time_of_download": post_data.get("time_of_download", ""),
        }

        if post_data.get("video_link"):
            metadata["video_link"] = post_data["video_link"]

        if post_data.get("image_links"):
            metadata["image_links"] = post_data["image_links"]

        if post_data.get("poll_content"):
            metadata["poll_content"] = post_data["poll_content"]

        return Document(page_content=text_content, metadata=metadata)
