from collections import defaultdict
import random
from collections import Counter
import torch
from sklearn.cluster import KMeans
import json


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def cluster_embeddings(embeddings: torch.Tensor, num_clusters: int) -> list[int]:
    """
    Clusters the sentence embeddings into groups using K-Means.

    Args:
        embeddings (torch.Tensor): A tensor of shape [n_samples, n_features] containing sentence embeddings.
        num_clusters (int): The number of clusters (groups) to form.

    Returns:
        list[int]: A list of group numbers (cluster assignments) for each sentence.
    """
    # Convert embeddings to NumPy array (KMeans requires NumPy arrays)
    embeddings_np = embeddings.cpu().numpy()

    # Initialize KMeans and fit the model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings_np)

    # Return the cluster labels (group numbers)
    return kmeans.labels_.tolist()


def add_group_numbers_to_jsonl(original_file_path: str, new_file_path: str, group_numbers: list[int]):
    """
    Adds the group number to each item in the original JSONL file and writes the updated data to a new file.

    Args:
        original_file_path (str): The path to the original JSONL file.
        new_file_path (str): The path to the new JSONL file to save the updated data.
        group_numbers (list[int]): A list of group numbers corresponding to each entry in the original file.

    Raises:
        AssertionError: If the length of group_numbers does not match the number of entries in the original file.
    """
    # Read the original JSONL file
    with open(original_file_path, 'r', encoding='utf-8') as file:
        original_data = [json.loads(line.strip()) for line in file]

    # Assert the length of group_numbers matches the number of entries
    assert len(original_data) == len(
        group_numbers), "The length of group_numbers must match the number of items in the original file."

    # Add the group_number to each item
    for i, data in enumerate(original_data):
        # Add the group_number as a new key
        data['group_number'] = group_numbers[i]

    # Write the updated data to the new file
    with open(new_file_path, 'w', encoding='utf-8') as file:
        for data in original_data:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')  # Write each JSON object on a new line


def count_items_in_groups(group_numbers: list[int]) -> dict[int, int]:
    """
    Counts how many items exist in each group based on the group_numbers list.

    Args:
        group_numbers (list[int]): A list of group numbers.

    Returns:
        dict[int, int]: A dictionary where the keys are group numbers and values are the counts of items in each group.
    """
    # Use Counter to count the occurrences of each group number
    group_counts = Counter(group_numbers)

    return dict(group_counts)


def sample_by_group(items: list[dict], group_numbers: list[int], samples_per_group: list[int]) -> list[list[dict]]:
    """
    Samples a specified number of items from each group. Ensures each group has enough items to sample.

    Args:
        items (list[dict]): A list of QA pairs, where each item is a dictionary.
        group_numbers (list[int]): A list of group numbers corresponding to each item.
        samples_per_group (list[int]): A list specifying the number of samples to draw from each group.

    Returns:
        list[list[dict]]: A list of lists, where each inner list contains the sampled items for the corresponding group.

    Raises:
        ValueError: If a group doesn't have enough items to sample.
    """
    # Group items by their group numbers
    grouped_items = defaultdict(list)
    for item, group in zip(items, group_numbers):
        grouped_items[group].append(item)

    # List to store the sampled items for each group
    sampled_items = []

    # Loop through each group's required sample size and sample items
    for group, sample_count in enumerate(samples_per_group):
        if group not in grouped_items:
            raise ValueError(f"Group {group} is missing from the dataset.")

        group_items = grouped_items[group]

        # Ensure the group has enough items to sample
        if len(group_items) < sample_count:
            raise ValueError(
                f"Group {group} does not have enough items to sample. Available: {len(group_items)}, Requested: {sample_count}")

        # Randomly sample the required number of items
        sampled_items.append(random.sample(group_items, sample_count))

    return sampled_items
