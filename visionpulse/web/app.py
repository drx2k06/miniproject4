"""Streamlit Cognitive Web Studio for VisionPulse."""

import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from visionpulse import __author__, __version__
from visionpulse.benchmarks.test_dataset import SampleDataset
from visionpulse.core.overlay import HUDOverlay
from visionpulse.core.pipeline import VisionPipeline
from visionpulse.models.base import TelemetryData


@st.cache_resource
def get_pipeline() -> VisionPipeline:
    """Initialize and cache the pipeline singleton."""
    pipe = VisionPipeline()
    # Preload models
    pipe.classifier.load_model()
    pipe.detector.load_model()
    return pipe


def main() -> None:
    st.set_page_config(
        page_title="VisionPulse Studio",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 12px 16px;
            border: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">👁️ VisionPulse Cognitive Studio</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Dual-Engine Vision Perception Suite v{__version__} | Developed by {__author__}</div>',
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()
    dataset = SampleDataset()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Perception Controls")

        mode = st.selectbox(
            "Inference Mode",
            options=["detection", "classification", "fusion"],
            format_func=lambda m: {
                "detection": "🎯 YOLOv8 Object Detection",
                "classification": "🧠 MobileNetV2 Classification",
                "fusion": "⚡ Dual-Engine Fusion",
            }[m],
        )
        pipeline.set_mode(mode)

        st.divider()

        conf_thresh = st.slider("Detection Confidence Threshold", 0.05, 0.95, 0.35, 0.05)
        top_k = st.slider("Classification Top-K Predictions", 1, 10, 5)

        st.divider()
        st.subheader("📥 Input Source")
        source_mode = st.radio(
            "Select Input Source",
            options=["Sample Library", "Upload Image", "Synthetic Stream Frame"],
        )

    # Source Ingestion
    input_bgr: np.ndarray = None

    if source_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            st.info("👆 Please upload an image file or switch to 'Sample Library'.")
            return

    elif source_mode == "Sample Library":
        samples = dataset.get_test_samples()
        sample_names = [name for name, _, _ in samples]
        selected_name = st.selectbox("Choose a sample scene:", sample_names)
        for name, img, _ in samples:
            if name == selected_name:
                input_bgr = img.copy()
                break

    else:  # Synthetic Stream Frame
        from visionpulse.core.stream import StreamSource
        with StreamSource(source="synthetic") as stream:
            _, input_bgr = stream.read()

    if input_bgr is None:
        st.warning("No input frame available.")
        return

    # Execute Inference Pipeline
    annotated_bgr, result, telemetry = pipeline.process_frame(
        input_bgr,
        mode=mode,
        conf_threshold=conf_thresh,
    )

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Inference Latency", value=f"{result.latency_ms:.1f} ms")
    with col2:
        st.metric(label="Throughput Estimate", value=f"{telemetry.fps:.1f} FPS")
    with col3:
        if mode in ("detection", "fusion"):
            st.metric(label="Detected Entities", value=f"{len(result.boxes)} objects")
        else:
            st.metric(label="Top Prediction", value=result.classifications[0].class_name if result.classifications else "N/A")
    with col4:
        st.metric(label="Active Device", value=str(result.device).upper())

    # Main Visual Display & Analysis Tabs
    tab_view, tab_telemetry, tab_compare = st.tabs(["🖼️ Visual Perception", "📊 Structured Telemetry", "🔄 Dual-Engine Comparison"])

    with tab_view:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Original Input")
            rgb_orig = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb_orig, use_container_width=True)

        with c2:
            st.subheader(f"Annotated ({result.mode.title()})")
            rgb_annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, use_container_width=True)

        if result.context_summary:
            st.info(f"**Cognitive Insight:** {result.context_summary}")

    with tab_telemetry:
        if mode in ("detection", "fusion") and result.boxes:
            st.subheader("Detected Objects")
            df_boxes = pd.DataFrame([b.model_dump() for b in result.boxes])
            st.dataframe(df_boxes[["class_name", "confidence", "x1", "y1", "x2", "y2"]], use_container_width=True)

            # Class count distribution chart
            if result.counts:
                df_counts = pd.DataFrame(list(result.counts.items()), columns=["Entity", "Count"])
                fig = px.bar(
                    df_counts,
                    x="Entity",
                    y="Count",
                    color="Entity",
                    title="Detected Entity Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

        if mode in ("classification", "fusion") and result.classifications:
            st.subheader("Classification Probabilities")
            df_clf = pd.DataFrame([c.model_dump() for c in result.classifications[:top_k]])
            df_clf["confidence_pct"] = df_clf["confidence"] * 100.0

            fig_clf = px.bar(
                df_clf,
                x="confidence_pct",
                y="class_name",
                orientation="h",
                labels={"confidence_pct": "Confidence (%)", "class_name": "Class"},
                title="Top-K Class Probabilities",
                color="confidence_pct",
                color_continuous_scale="Blues",
            )
            fig_clf.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_clf, use_container_width=True)

        st.subheader("Raw JSON Telemetry")
        st.json(result.model_dump())

    with tab_compare:
        st.subheader("Side-by-Side Model Comparison (MobileNetV2 vs. YOLOv8n)")
        clf_res = pipeline.classifier.predict(input_bgr, top_k=3)
        det_res = pipeline.detector.predict(input_bgr, conf_threshold=conf_thresh)

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("#### 🧠 MobileNetV2 (Classification)")
            st.write(f"**Latency:** {clf_res.latency_ms:.1f} ms")
            if clf_res.classifications:
                for idx, c in enumerate(clf_res.classifications[:3]):
                    st.write(f"**#{idx+1} {c.class_name}:** {c.confidence:.1%}")
            st.caption("Classifies the primary global semantic subject of the image.")

        with comp_col2:
            st.markdown("#### 🎯 YOLOv8n (Object Detection)")
            st.write(f"**Latency:** {det_res.latency_ms:.1f} ms")
            st.write(f"**Total Objects:** {len(det_res.boxes)}")
            if det_res.counts:
                st.write("**Counts:**", ", ".join([f"{k}: {v}" for k, v in det_res.counts.items()]))
            st.caption("Localizes multiple foreground objects with coordinates.")


if __name__ == "__main__":
    main()
