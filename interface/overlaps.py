import streamlit as st
import pandas as pd
from annotated_text import annotated_text
from sumtool.storage import get_summaries


def st_annotate_highlights(text, highlights):
    highlighted_text = text
    for highlight in highlights:
        # TODO should this be normalized to handle i.e. lower-case matches?
        highlighted_text = highlighted_text.replace(
            highlight, "--hl--hl>>" + highlight + "<<hl--hl--"
        )

    text_with_highlights = []
    for fragment in highlighted_text.split("--hl--"):
        if fragment.startswith("hl>>") and fragment.endswith("<<hl"):
            sanitized_fragment = fragment.replace("hl>>", "").replace("<<hl", "")
            text_with_highlights.append((sanitized_fragment, "highlight"))
        else:
            text_with_highlights.append(fragment)
    highlighted_text.split

    return annotated_text(*text_with_highlights)


def load_newsroom_data():
    all_data = get_summaries('newsroom', 'gold')

    return [
        {
            "id": index,
            "url": data['metadata']['url'],
            "source": data['metadata']['source'],
            "summary": data['summary'],
            "highlights": data['metadata']['supported_summary_entities'],
            "density_bin": data['metadata']['density_bin'],
            "compression_bin": data['metadata']['compression_bin'],
            "coverage_bin": data['metadata']['compression_bin'],
        } for index, data in all_data.items()
        if data['metadata']['density_bin'] in ['abstractive', 'mixed']
    ]

def load_xsum_data():
    all_data = get_summaries('xsum', 'gold')

    return [
        {
            "id": index,
            "source": data['metadata']['source'],
            "summary": data['summary'],
            "highlights": data['metadata']['supported_summary_entities'],
        } for index, data in all_data.items()
    ]


def render_overlaps():
    st.title("Summary Overlaps")
    summary_with_overlaps = load_newsroom_data()
    for i, data in enumerate(summary_with_overlaps):
        st.subheader(f"Summary '{data['id']}'")
        st.table(
            pd.DataFrame(
                {
                    "url": [data['url']],
                    "density_bin": [data['density_bin']],
                    "compression_bin": [data['compression_bin']],
                    "coverage_bin": [data['coverage_bin']],
                }).iloc[0]
        )

        st.write("**Ground Truth Summary:**")
        st_annotate_highlights(data["summary"], data["highlights"])
        st.write("**Source Document:**")
        st_annotate_highlights(data["source"], data["highlights"])


if __name__ == "__main__":
    render_overlaps()
