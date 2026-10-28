"""Configuration management for VisionPulse using Pydantic Settings."""

import os
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Model-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONPULSE_",
        env_file=".env",
        extra="ignore",
    )

    device: str = Field(
        default="auto",
        description="Inference device: 'auto', 'cpu', 'cuda', 'mps'",
    )
    yolo_model: str = Field(
        default="yolov8n.pt",
        description="YOLO model name or checkpoint path",
    )
    yolo_conf_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for YOLO object detections",
    )
    yolo_iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IOU threshold for Non-Maximum Suppression (NMS)",
    )
    mobilenet_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Top-K predicted classes for MobileNetV2",
    )


class StreamConfig(BaseSettings):
    """Video and image stream configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONPULSE_",
        env_file=".env",
        extra="ignore",
    )

    camera_index: int = Field(default=0, ge=0)
    frame_width: int = Field(default=1280, ge=160)
    frame_height: int = Field(default=720, ge=120)
    fps_target: int = Field(default=30, ge=1, le=120)


class ExportConfig(BaseSettings):
    """Output and export configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONPULSE_",
        env_file=".env",
        extra="ignore",
    )

    export_dir: str = Field(default="./output")
    save_annotated_frames: bool = Field(default=False)
    save_telemetry_json: bool = Field(default=True)
    video_codec: str = Field(default="mp4v")


class VisionConfig(BaseSettings):
    """Root configuration object for VisionPulse."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONPULSE_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = Field(default="VisionPulse")
    env: Literal["development", "production", "test"] = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    models: ModelConfig = Field(default_factory=ModelConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    web_port: int = Field(default=8501, ge=1024, le=65535)


def get_config() -> VisionConfig:
    """Load and return active VisionConfig singleton."""
    return VisionConfig()
