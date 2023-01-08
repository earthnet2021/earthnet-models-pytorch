async function runInceptionV2() {
    // Creat the session and load the pre-trained model
    const session = new onnx.InferenceSession({ backendHint: 'webgl' });
    await session.loadModel("./inception_v2/model.onnx");
  
    // Load image.
    const imageLoader = new ImageLoader(imageSize, imageSize);
    const imageData = await imageLoader.getImageData('./cat.jpg');
  
    // Resize the Input image to 1*3*224*224.
    const width = imageSize;
    const height = imageSize;
    const preprocessedData = preprocess(imageData.data, width, height);
  
    const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 3, 244, 244]);
    
    // Run model with Tensor inputs and get the result.
    const outputMap = await session.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;
  
    // Render the results in html.
    printMatches(outputData);
  }