import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path("reports/charts")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def generate_chart(
    chart_spec,
    index: int
):
    path = (
        OUTPUT_DIR /
        f"chart_{index}.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    if chart_spec["chart_type"] == "bar":

        plt.bar(
            chart_spec["labels"],
            chart_spec["values"]
        )

    elif chart_spec["chart_type"] == "line":

        plt.plot(
            chart_spec["labels"],
            chart_spec["values"]
        )

    elif chart_spec["chart_type"] == "pie":

        plt.pie(
            chart_spec["values"],
            labels=chart_spec["labels"]
        )

    plt.title(
        chart_spec["title"]
    )

    plt.xlabel(
        chart_spec["x_label"]
    )

    plt.ylabel(
        chart_spec["y_label"]
    )

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return str(path)