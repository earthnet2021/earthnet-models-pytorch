// https://rekoil.io/blog/onnxruntime-web-tutorial/

const ort = require('onnxruntime-web');

// Model inference
async function run() {
  try {
    // create a new session and load the AlexNet model.
    const session = await ort.InferenceSession.create("./alexnet.onnx") // "experiments/en23/context-convlstm/baseline_RGBNR.onnx")

    // prepare dummy input data
    const dims = [1, 3, 224, 224] // [35, 30, 128, 128]
    const size = dims[0] * dims[1] * dims[2] * dims[3]
    const inputData = Float32Array.from({ length: size }, () => Math.random())

    // prepare feeds. use model input names as keys.
    const feeds = { input1: new ort.Tensor("float32", inputData, dims) }

    // feed inputs and run
    const results = await session.run(feeds)
    console.log(results.output1.data)
  } catch (e) {
    console.log(e)
  }
}
run()