from pathlib import Path

import pandas as pd


DATASETS_DIR = "datasets"


class DataExtractor:
    def __init__(
        self,
        dataset_name,
        dataset_file,
        class_labels,
        columns,
        class_column,
        num_per_class,
    ):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.class_labels = class_labels
        self.num_classes = len(self.class_labels)
        self.num_per_class = num_per_class
        self.columns = columns
        self.class_column = class_column
        assert self.class_column in self.columns

    def _sample_class(self, label):
        json_reader = pd.read_json(
            self.dataset_file, orient="records", lines=True, chunksize=10_000,
        )
        df = pd.DataFrame()
        actual_sampled = 0
        for chunk in json_reader:
            chunk = chunk.drop(
                columns=[c for c in chunk.columns if c not in self.columns]
            )
            s = chunk[chunk[self.class_column] == label]
            class_count = s[self.class_column].value_counts()[label]
            if class_count < 1:
                # There arent elements of the class in the current chunk
                continue
            sample_size = min(self.num_per_class - actual_sampled, class_count)
            sample = s.sample(sample_size, random_state=20)
            assert (
                sample[sample[self.class_column] == label].count()
                == sample.count()
            ).all()
            assert (
                sample[sample[self.class_column] == label][
                    self.class_column
                ].value_counts()
                == [sample_size]
            ).all()
            df = df.append(sample)
            actual_sampled += sample_size
            if actual_sampled >= self.num_per_class:
                break
        else:
            raise ValueError(
                f"Can't find {self.num_per_class} samples for label {label}"
            )
        return df

    def setup_data(self):
        """
        Create a csv file DATASETS_DIR/<dataset_name>.csv
        """
        df = pd.DataFrame()
        for label in self.class_labels:
            df = df.append(self._sample_class(label))
        df = df.sample(frac=1, random_state=20)  # shuffle the DataFrame
        assert (
            df[self.class_column].value_counts(sort=False)
            == [self.num_per_class] * self.num_classes
        ).all()
        df.to_csv(f"{DATASETS_DIR}/{self.dataset_name}.csv", index=False)


def main():
    # DataExtractor(
    #    dataset_name="yelp",
    #    dataset_file=Path(DATASETS_DIR) / "yelp.json",
    #    class_labels=(1, 2, 3, 4, 5),
    #    num_per_class=1_569_265 // 5,
    #    columns=("text", "stars"),
    #    class_column="stars",
    # ).setup_data()

    DataExtractor(
        dataset_name="amazon",
        dataset_file=Path(DATASETS_DIR) / "amazon_books.json.gz",
        class_labels=(1, 2, 3, 4, 5),
        num_per_class=3_650_000 // 5,
        columns=("reviewText", "overall"),
        class_column="overall",
    ).setup_data()


if __name__ == "__main__":
    main()
