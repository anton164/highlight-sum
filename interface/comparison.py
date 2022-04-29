import streamlit as st
from annotated_text import annotated_text
from sumtool.storage import get_models, get_summaries, get_summary_metrics
from data_utils import load_xsum_dict

st.set_page_config(layout="wide")


def annotation_color(type):
    if type == "extrinsic":
        return "yellow"
    if type == "contained":
        return "lightblue"


def st_annotate_highlights(text, highlights_intrinsic, highlights_extrinsic):
    highlighted_text = text
    for highlight in set(highlights_intrinsic + highlights_extrinsic):
        # TODO should this be normalized to handle i.e. lower-case matches?
        highlighted_text = highlighted_text.replace(
            highlight, "--hl--hl>>" + highlight + "<<hl--hl--"
        )

    text_with_highlights = []
    for fragment in highlighted_text.split("--hl--"):
        if fragment.startswith("hl>>") and fragment.endswith("<<hl"):
            sanitized_fragment = fragment.replace("hl>>", "").replace("<<hl", "")
            ann_type = (
                "extrinsic" 
                if sanitized_fragment in highlights_extrinsic
                else "contained"
            )
            text_with_highlights.append(
                (
                    sanitized_fragment,
                    ann_type,
                    annotation_color(ann_type)
                )
            )
        else:
            text_with_highlights.append(fragment)

    return annotated_text(*text_with_highlights)


def metadata_stats(metadata):
    intrinsic = sum([1 for x in metadata.values() if len(x["entities_in_source"]) > 0])
    extrinsic = sum([1 for x in metadata.values() if len(x["entities_not_in_source"]) > 0])
    total = len(metadata)

    return {
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "total": total
    }

def render_model_comparison():
    st.title("Model Comparison")
    xsum_test = load_xsum_dict("test")
    gt_metadata = get_summary_metrics("xsum", "gold")
    available_models = get_models("xsum")

    model_1 = st.selectbox("Model 1", options=available_models, index=0)
    model_1_summaries = get_summaries("xsum", model_1)
    model_1_metadata = get_summary_metrics("xsum", model_1)
    model_2 = st.selectbox("Model 2", options=available_models, index=1)
    model_2_summaries = get_summaries("xsum", model_2)
    model_2_metadata = get_summary_metrics("xsum", model_2)

    model_1_keys = [
        key for key, val
        in model_1_metadata.items()
        if len(val["entities_not_in_source"]) > 0
    ]

    model_2_keys = [
        key for key, val
        in model_2_metadata.items()
        if len(val["entities_not_in_source"]) == 0
    ]

    overlapping_keys = set(
        model_1_keys
    ).intersection(
        set(model_2_keys)
    )

    selected_id = str(
        st.selectbox(
            "Select entry by bbcid", 
            options=list(overlapping_keys)
        )
    )

    st.subheader("Stats")
    gt_stats = metadata_stats(gt_metadata)
    model1_stats = metadata_stats(model_1_metadata)
    model2_stats = metadata_stats(model_2_metadata)
    
    st.write(
f"""
    - Ground truth sums with intrinsic: {gt_stats['intrinsic']}, extrinsic: {gt_stats['extrinsic']} out of {gt_stats['total']}   
    - '{model_1}' sums with intrinsic: {model1_stats['intrinsic']}, extrinsic: {model1_stats['extrinsic']} out of {model1_stats['total']}      
    - '{model_2}' sums with intrinsic: {model2_stats['intrinsic']}, extrinsic: {model2_stats['extrinsic']} out of {model2_stats['total']}   
"""
    )

    col1, col2, col3 = st.columns(3)
    xsum_example = xsum_test[selected_id]

    with col1:
        st.subheader("Ground Truth Summary")
        st_annotate_highlights(
            xsum_test[selected_id]["summary"],
            gt_metadata[selected_id]["entities_in_source"],
            gt_metadata[selected_id]["entities_not_in_source"],
        )
        st.write(gt_metadata[selected_id])

    with col2:
        st.subheader(model_1)
        if selected_id not in model_1_summaries:
            st.write("Summary missing")
        else:
            st_annotate_highlights(
                model_1_summaries[selected_id]["summary"],
                model_1_metadata[selected_id]["entities_in_source"],
                model_1_metadata[selected_id]["entities_not_in_source"],
            )
            st.write(model_1_metadata[selected_id])

    with col3:
        st.subheader(f"{model_2}")
        if selected_id not in model_2_summaries:
            st.write("Summary missing")
        else:
            st_annotate_highlights(
                model_2_summaries[selected_id]["summary"],
                model_2_metadata[selected_id]["entities_in_source"],
                model_2_metadata[selected_id]["entities_not_in_source"],
            )
            st.write(model_2_metadata[selected_id])

    st.subheader("Source document")
    st.text(xsum_example["document"])


if __name__ == "__main__":
    render_model_comparison()
