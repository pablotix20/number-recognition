// import * as tf from '@tensorflow/tfjs';
const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

async function run() {
  model = await tf.loadGraphModel('model_js/model.json');
  console.log(model)
  demosSection.classList.remove('invisible');
  // enableCam()
}

run();

// Check if webcam access is supported.
function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia);
  }
  
  // If webcam supported, add event listener to button for when user
  // wants to activate it to call enableCam function which we will 
  // define in the next step.
  if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }

var webcam
  // Enable the live webcam view and start classification.
function enableCam(event) {
    console.log('click')
    // Only continue if the COCO-SSD has finished loading.
    if (!model) {
      return;
    }
    
    // Hide the button once clicked.
    event.target.classList.add('removed');  
    
    // getUsermedia parameters to force video but not audio.
    // const constraints = {
    //   video: { facingMode: "environment",width:288,height:288 }
    // };
    const constraints = {
      video: { width:288,height:288 }
    };
  
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then( async function(stream) {
      console.log(stream)
      video.width = 288;
      video.height = 288;
      video.srcObject = stream;

      webcam = await tf.data.webcam(video, {
        resizeWidth: 288,
        resizeHeight: 288,
      });

      // console.log(webcam)

      video.addEventListener('loadeddata', predictWebcam);
    });
  }

  // Placeholder function for next step.
function predictWebcam() {
}

// Store the resulting model in the global scope of our app.
var model = undefined;

var children = [];
var canvas =[];
for(let i=0;i<11;i++){
  canvas.push(document.getElementById(`canvas-${i}`))
}

async function predictWebcam() {
  tf.engine().startScope()

  const img = await webcam.capture();

  reshaped = img
    .mean(2)
    .toFloat()
    .expandDims(0)
    .expandDims(-1)
  
  reshaped = reshaped.div(255)
  // reshaped.print()
  // Pass the data to the loaded mobilenet model.
  const result = await model.predict(reshaped).reshape([144,144,11]);
  console.log(result)
  const channels = tf.split(result,11,2)

  for(let i=0;i<11;i++){
    tf.browser.toPixels(channels[i],canvas[i])
  }
  
  // result.dispose()
  // img.dispose()
  // reshaped.dispose()


  window.requestAnimationFrame(predictWebcam);

  tf.engine().endScope()

  return
  model.detect(video).then(function (predictions) {
    // Remove any highlighting we did previous frame.
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i]);
    }
    children.splice(0);
    
    // Now lets loop through predictions and draw them to the live view if
    // they have a high confidence score.
    for (let n = 0; n < predictions.length; n++) {
      // If we are over 66% sure we are sure we classified it right, draw it!
      if (predictions[n].score > 0.66) {
        const p = document.createElement('p');
        p.innerText = predictions[n].class  + ' - with ' 
            + Math.round(parseFloat(predictions[n].score) * 100) 
            + '% confidence.';
        p.style = 'margin-left: ' + predictions[n].bbox[0] + 'px; margin-top: '
            + (predictions[n].bbox[1] - 10) + 'px; width: ' 
            + (predictions[n].bbox[2] - 10) + 'px; top: 0; left: 0;';

        const highlighter = document.createElement('div');
        highlighter.setAttribute('class', 'highlighter');
        highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: '
            + predictions[n].bbox[1] + 'px; width: ' 
            + predictions[n].bbox[2] + 'px; height: '
            + predictions[n].bbox[3] + 'px;';

        liveView.appendChild(highlighter);
        liveView.appendChild(p);
        children.push(highlighter);
        children.push(p);
      }
    }
    
    // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam);
  });
}