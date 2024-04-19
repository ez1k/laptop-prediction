"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline


from .nodes import transform_data, save_data

def create_pipeline(** kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=transform_data,
                inputs="laptops",
                outputs="laptops_for_model",
                name="transform_laptops_node"
            )
        ]
    )