import os.path

import pandas as pd

from slugify import slugify


def create_markdown_from_row(data: dict) -> str:
    markdown_template = (
        "# {title}\n\n"
        "## Location\n"
        "### Country\n"
        "{country}\n"
        "### Province\n"
        "{province}\n\n"
        "## Description\n"
        "{description}\n\n"
    )

    details_cols = [
        "designation",
        "points",
        "price",
        "region_1",
        "region_2",
        "taster_name",
        "taster_twitter_handle",
        "variety",
        "winery",
    ]
    details = "## Details\n"
    for col in details_cols:
        if not pd.isna(data[col]):
            details += f"### {col.capitalize()}\n{data[col]}\n"

    markdown_document = markdown_template.format(**data)
    markdown_document += details
    return markdown_document


if __name__ == "__main__":
    df = pd.read_csv("./winemag-data-130k-v2.csv")
    output_dir = "./input"
    n = len(df)
    for idx, row in df.iterrows():
        filename = f"{slugify(row['title'])}.txt"
        md_text = create_markdown_from_row(row.to_dict())
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(md_text)
        print(f"{filename} ({idx}/{n})")
