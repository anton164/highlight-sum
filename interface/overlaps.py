import streamlit as st
from annotated_text import annotated_text


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


def load_data():

    return [
        {
            "id": "dummy-train-0",
            "source": """
The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town. First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit. However, she said more preventative work could have been carried out to ensure the retaining wall did not fail. "It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten," she said. "That may not be true but it is perhaps my perspective over the last few days. "Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?" Meanwhile, a flood alert remains in place across the Borders because of the constant rain. Peebles was badly hit by problems, sparking calls to introduce more defences in the area. Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs. The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand. He said it was important to get the flood protection plan right but backed calls to speed up the process. "I was quite taken aback by the amount of damage that has been done," he said. "Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses." He said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans. Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.
""",
            "summary": "Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.",
            "highlights": ["Scottish Borders", "Dumfries", "Galloway"],
        }
    ]


def render_overlaps():
    st.title("Summary Overlaps")
    summary_with_overlaps = load_data()
    for i, data in enumerate(summary_with_overlaps):
        st.subheader(f"Summary '{data['id']}'")
        st.write("**Ground Truth Summary:**")
        st_annotate_highlights(data["summary"], data["highlights"])
        st.write("**Source Document:**")
        st_annotate_highlights(data["source"], data["highlights"])


if __name__ == "__main__":
    render_overlaps()
