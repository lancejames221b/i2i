"""Tests for i2i multimodal support."""

import base64
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch, MagicMock

from i2i.schema import (
    Message,
    MessageType,
    ContentType,
    Attachment,
)
from i2i.providers import model_supports_vision, VISION_CAPABLE_MODELS


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_types_defined(self):
        """All expected content types should be defined."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.AUDIO == "audio"
        assert ContentType.VIDEO == "video"
        assert ContentType.DOCUMENT == "document"


class TestAttachment:
    """Tests for Attachment model."""

    def test_create_attachment_with_data(self):
        """Create an attachment with base64 data."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            mime_type="image/png",
        )
        assert attachment.content_type == ContentType.IMAGE
        assert attachment.data is not None
        assert attachment.mime_type == "image/png"

    def test_create_attachment_with_url(self):
        """Create an attachment with URL reference."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/image.png",
            description="Example image",
        )
        assert attachment.content_type == ContentType.IMAGE
        assert attachment.url == "https://example.com/image.png"
        assert attachment.description == "Example image"

    def test_attachment_requires_data_or_url(self):
        """Attachment must have either data or url."""
        with pytest.raises(ValueError, match="Either 'data'.*or 'url' must be provided"):
            Attachment(content_type=ContentType.IMAGE)

    def test_attachment_with_description(self):
        """Attachment can have description for fallback."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/chart.png",
            description="A bar chart showing sales data for Q4 2024",
        )
        assert attachment.description == "A bar chart showing sales data for Q4 2024"

    def test_get_base64_data_from_data(self):
        """get_base64_data returns data field if present."""
        data = "SGVsbG8gV29ybGQ="
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            data=data,
        )
        assert attachment.get_base64_data() == data

    def test_get_base64_data_from_data_uri(self):
        """get_base64_data extracts from data URI."""
        data = "SGVsbG8gV29ybGQ="
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url=f"data:image/png;base64,{data}",
        )
        assert attachment.get_base64_data() == data

    def test_get_base64_data_returns_none_for_http_url(self):
        """get_base64_data returns None for http URLs."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/image.png",
        )
        assert attachment.get_base64_data() is None

    def test_infer_mime_type_from_field(self):
        """infer_mime_type uses mime_type field first."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/image",
            mime_type="image/webp",
        )
        assert attachment.infer_mime_type() == "image/webp"

    def test_infer_mime_type_from_data_uri(self):
        """infer_mime_type extracts from data URI."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="data:image/jpeg;base64,abc123",
        )
        assert attachment.infer_mime_type() == "image/jpeg"

    def test_infer_mime_type_from_filename(self):
        """infer_mime_type guesses from filename."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/photo.jpg",
            filename="photo.jpg",
        )
        assert attachment.infer_mime_type() == "image/jpeg"

    def test_from_url_classmethod(self):
        """Attachment.from_url creates URL-based attachment."""
        attachment = Attachment.from_url(
            "https://example.com/diagram.png",
            content_type=ContentType.IMAGE,
            description="System architecture diagram",
        )
        assert attachment.url == "https://example.com/diagram.png"
        assert attachment.content_type == ContentType.IMAGE
        assert attachment.description == "System architecture diagram"

    def test_from_file_classmethod(self, tmp_path):
        """Attachment.from_file creates file-based attachment."""
        # Create a temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        attachment = Attachment.from_file(str(test_file), description="Test file")
        assert attachment.content_type == ContentType.DOCUMENT
        assert attachment.filename == "test.txt"
        assert attachment.data is not None
        assert attachment.description == "Test file"

    def test_from_file_image(self, tmp_path):
        """Attachment.from_file detects image content type."""
        # Create a minimal PNG
        test_file = tmp_path / "test.png"
        # Minimal 1x1 transparent PNG
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        test_file.write_bytes(png_data)

        attachment = Attachment.from_file(str(test_file))
        assert attachment.content_type == ContentType.IMAGE
        assert attachment.mime_type == "image/png"

    def test_from_file_not_found(self):
        """Attachment.from_file raises for missing file."""
        with pytest.raises(FileNotFoundError):
            Attachment.from_file("/nonexistent/path/image.png")


class TestMessageMultimodal:
    """Tests for Message multimodal support."""

    def test_message_has_attachments_field(self):
        """Message should have attachments field."""
        message = Message(type=MessageType.QUERY, content="Test")
        assert hasattr(message, "attachments")
        assert isinstance(message.attachments, list)

    def test_message_empty_attachments_by_default(self):
        """Message attachments should be empty by default."""
        message = Message(type=MessageType.QUERY, content="Test")
        assert len(message.attachments) == 0

    def test_message_with_attachments(self):
        """Message can be created with attachments."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/img.png",
        )
        message = Message(
            type=MessageType.QUERY,
            content="Describe this image",
            attachments=[attachment],
        )
        assert len(message.attachments) == 1
        assert message.attachments[0].content_type == ContentType.IMAGE

    def test_has_attachments_true(self):
        """has_attachments returns True when attachments exist."""
        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/img.png",
        )
        message = Message(
            type=MessageType.QUERY,
            content="Test",
            attachments=[attachment],
        )
        assert message.has_attachments() is True

    def test_has_attachments_false(self):
        """has_attachments returns False when no attachments."""
        message = Message(type=MessageType.QUERY, content="Test")
        assert message.has_attachments() is False

    def test_get_attachments_by_type(self):
        """get_attachments_by_type filters by content type."""
        attachments = [
            Attachment(content_type=ContentType.IMAGE, url="https://example.com/1.png"),
            Attachment(content_type=ContentType.AUDIO, url="https://example.com/1.mp3"),
            Attachment(content_type=ContentType.IMAGE, url="https://example.com/2.png"),
        ]
        message = Message(
            type=MessageType.QUERY,
            content="Test",
            attachments=attachments,
        )
        images = message.get_attachments_by_type(ContentType.IMAGE)
        assert len(images) == 2
        audio = message.get_attachments_by_type(ContentType.AUDIO)
        assert len(audio) == 1

    def test_get_image_attachments(self):
        """get_image_attachments is shorthand for get_attachments_by_type(IMAGE)."""
        attachments = [
            Attachment(content_type=ContentType.IMAGE, url="https://example.com/1.png"),
            Attachment(content_type=ContentType.VIDEO, url="https://example.com/1.mp4"),
        ]
        message = Message(
            type=MessageType.QUERY,
            content="Test",
            attachments=attachments,
        )
        assert len(message.get_image_attachments()) == 1

    def test_add_attachment(self):
        """add_attachment appends to attachments list."""
        message = Message(type=MessageType.QUERY, content="Test")
        assert len(message.attachments) == 0

        attachment = Attachment(
            content_type=ContentType.IMAGE,
            url="https://example.com/img.png",
        )
        message.add_attachment(attachment)
        assert len(message.attachments) == 1

    def test_get_text_with_descriptions_no_attachments(self):
        """get_text_with_descriptions returns content when no attachments."""
        message = Message(type=MessageType.QUERY, content="Test content")
        assert message.get_text_with_descriptions() == "Test content"

    def test_get_text_with_descriptions_with_attachments(self):
        """get_text_with_descriptions appends attachment descriptions."""
        attachments = [
            Attachment(
                content_type=ContentType.IMAGE,
                url="https://example.com/chart.png",
                description="Sales chart for Q4",
            ),
            Attachment(
                content_type=ContentType.IMAGE,
                url="https://example.com/logo.png",
            ),
        ]
        message = Message(
            type=MessageType.QUERY,
            content="Analyze these images",
            attachments=attachments,
        )
        result = message.get_text_with_descriptions()
        assert "Analyze these images" in result
        assert "[Attachment 1: Sales chart for Q4]" in result
        assert "[Attachment 2: image]" in result  # Falls back to content type


class TestVisionCapableModels:
    """Tests for vision-capable model detection."""

    def test_vision_capable_models_set_exists(self):
        """VISION_CAPABLE_MODELS set should exist."""
        assert isinstance(VISION_CAPABLE_MODELS, set)
        assert len(VISION_CAPABLE_MODELS) > 0

    def test_gpt4o_is_vision_capable(self):
        """GPT-4o models should support vision."""
        assert "gpt-4o" in VISION_CAPABLE_MODELS
        assert "gpt-4o-mini" in VISION_CAPABLE_MODELS

    def test_claude3_is_vision_capable(self):
        """Claude 3 models should support vision."""
        assert "claude-3-opus-20240229" in VISION_CAPABLE_MODELS
        assert "claude-sonnet-4-5-20250929" in VISION_CAPABLE_MODELS

    def test_gemini_is_vision_capable(self):
        """Gemini models should support vision."""
        assert "gemini-1.5-pro" in VISION_CAPABLE_MODELS
        assert "gemini-3-flash-preview" in VISION_CAPABLE_MODELS

    def test_model_supports_vision_true(self):
        """model_supports_vision returns True for vision models."""
        assert model_supports_vision("gpt-4o") is True
        assert model_supports_vision("claude-sonnet-4-5-20250929") is True

    def test_model_supports_vision_with_prefix(self):
        """model_supports_vision handles provider prefix."""
        assert model_supports_vision("openai/gpt-4o") is True
        assert model_supports_vision("anthropic/claude-sonnet-4-5-20250929") is True

    def test_model_supports_vision_false(self):
        """model_supports_vision returns False for non-vision models."""
        # Assuming text-only models like base GPT-3.5
        assert model_supports_vision("gpt-3.5-turbo") is False


class TestProviderMultimodalIntegration:
    """Integration tests for provider multimodal handling."""

    @pytest.fixture
    def image_attachment(self):
        """Create a sample image attachment."""
        return Attachment(
            content_type=ContentType.IMAGE,
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            mime_type="image/png",
            description="A 1x1 transparent PNG",
        )

    @pytest.fixture
    def multimodal_message(self, image_attachment):
        """Create a message with image attachment."""
        return Message(
            type=MessageType.QUERY,
            content="Describe this image",
            attachments=[image_attachment],
        )

    def test_message_with_image_serializes(self, multimodal_message):
        """Multimodal message should serialize to JSON."""
        json_str = multimodal_message.model_dump_json()
        assert "attachments" in json_str
        assert "image" in json_str

    def test_message_with_image_roundtrips(self, multimodal_message):
        """Multimodal message should roundtrip through JSON."""
        json_str = multimodal_message.model_dump_json()
        restored = Message.model_validate_json(json_str)
        assert len(restored.attachments) == 1
        assert restored.attachments[0].content_type == ContentType.IMAGE
