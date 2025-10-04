import argparse
from hart.clustering import Topic2VecClustering
from hart.utils.datasets import load_mjhq


def test_TTV():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjhq-meta-path",
        type=str,
        help="The path to MJHQ meta_data.json.",
        default="./MJHQ-30K/meta_data.json",
    )

    args = parser.parse_args()
    prompts = load_mjhq(args.get('mjhq-meta-path'))
    ttv = Topic2VecClustering()
    ttv.init_model(prompts)
    # Simplified version
    fig_simple = ttv.plot_interactive_with_centroids(
        file_name="simple_centroids_0_05_dist",
        save_path="./",
        show_fig=True
    )


if __name__ == "__main__":
    test_TTV()