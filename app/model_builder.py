# app/model_builder.py
import onnx
from onnx import helper, TensorProto


def build_add_model(path: str = "model.onnx"):
    # Create graph: C = A + B where A and B are float tensors of shape [1, N]
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, ['N'])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, ['N'])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, ['N'])

    node_def = helper.make_node(
        'Add',
        inputs=['A', 'B'],
        outputs=['C'],
    )

    graph_def = helper.make_graph(
        nodes=[node_def],
        name='add-graph',
        inputs=[A, B],
        outputs=[C],
    )

    model_def = helper.make_model(graph_def, producer_name='edge-onnx-sample')
    onnx.save(model_def, path)
    return path


if __name__ == '__main__':
    p = build_add_model()
    print(f'Wrote {p}')