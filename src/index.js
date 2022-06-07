import React, { useEffect, useState, createRef  } from "react";
import ReactDOM from "react-dom";
import MagicDropzone from "react-magic-dropzone";
import {useCamera} from './hooks/useCamera';
import "./styles.css";
const tf = require('@tensorflow/tfjs');

//const weights = '/web_model/model.json';
const weights = 'http://127.0.0.1:8080/model.json';
const names = ['contacts', 'isettings', 'imessage', 'whatsapp']
let videoPlaying = false;

let _lastRender = Date.now();

const  App = ()=>{
  
  const videoRef = createRef();
  const canvasRef = createRef();

  const [video, isCameraInitialised, playing, setPlaying, error] = useCamera(videoRef);
  const [ctx, setCtx] = useState();
  const [model, setModel] = useState();
  const [preview, setPreview] = useState();
  const [predictions, setPredictions] = useState([]); 
 
  let boxes_data;
  let scores_data; 
  let classes_data; 
  let valid_detections_data;

  useEffect(()=>{
    tf.loadGraphModel(weights).then(model => {
      setModel(model);
    });
  },[]);


  const renderPredictions = (c)=>{
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
      for (let i = 0; i < valid_detections_data; ++i){
        let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
      
        x1 *= c.width;
        x2 *= c.width;
        y1 *= c.height;
        y2 *= c.height;
        const width = x2 - x1;
        const height = y2 - y1;
        const klass = names[classes_data[i]];
        const score = scores_data[i].toFixed(2);
        ctx.fillStyle = "#00FFFF";
        // Draw the bounding box.
        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw the label background.
        ctx.fillStyle = "#00FFFF";
        const textWidth = ctx.measureText(klass /*+ ":" + score*/).width;
        const textHeight = parseInt(font, 10); // base 10
        ctx.fillRect(x1, y1, textWidth + 4, textHeight + 4);

      }
      for (let i = 0; i < valid_detections_data; ++i){
        let [x1, y1, , ] = boxes_data.slice(i * 4, (i + 1) * 4);
        x1 *= c.width;
        y1 *= c.height;
        const klass = names[classes_data[i]];
        const score = scores_data[i].toFixed(2);

        // Draw the text last to ensure it's on top.
        ctx.fillStyle = "#000000";
        //+ ":" + score  
        ctx.fillText(klass , x1, y1);
        
      }
  }

  const updateCanvas = ()=>{
    try{
      //if (ctx){
        
        let [modelWidth, modelHeight] = model.inputs[0].shape.slice(1, 3);
        const c = canvasRef.current;
        if (!c){
          return
        }
    
          
        //if want to control rate
        /*
        if (Date.now() - _lastRender < 200){
           ctx.drawImage(video,0,0,640,480);
           renderPredictions(c)
            window.requestAnimationFrame(updateCanvas); 
            return;
        }*/
     
        _lastRender = Date.now();

        const input = tf.tidy(() => {
           return tf.image.resizeBilinear(tf.browser.fromPixels(c), [modelWidth, modelHeight]).div(255.0).expandDims(0);
        });

         
         model.executeAsync(input).then(res => {
            // Font options.
            window.requestAnimationFrame(updateCanvas); 
            ctx.drawImage(video,0,0,640,480);
           
            const [boxes, scores, classes, valid_detections] = res;
           
            boxes_data = boxes.dataSync();
            scores_data = scores.dataSync();
            classes_data = classes.dataSync();
            valid_detections_data = valid_detections.dataSync()[0];
           
            tf.dispose(res)
            renderPredictions(c);
            
        });
       
    }catch(err){
      console.log(err);
    }
   
  }

  useEffect(() => {
 
    if(video) {
      const c = canvasRef.current;
      const ctx = c.getContext("2d");
      if (ctx){
        setCtx(ctx);
        updateCanvas();
      }
    }
  }, [video, canvasRef]);

  return (
      <div> {/*className="Dropzone-page">*/}

          <video ref={videoRef} style={{
                opacity: 1, /*videoopacity,*/
                position:"absolute",
                marginLeft: "auto",
                marginRight: "auto",
                left: 0,
                right: 0,
                textAlign: "center",
                zindex: 9,
                width: 640,
                height: 480,
                display:"none"
              }}
            />

        <canvas ref={canvasRef} id="canvas" width="640" height="480" />
        {/* model ? (
          <MagicDropzone
            className="Dropzone"
            accept="image/jpeg, image/png, .jpg, .jpeg, .png"
            multiple={false}
            onDrop={onDrop}
          >
            {preview ? (
              <img
                alt="upload preview"
                onLoad={onImageChange}
                className="Dropzone-img"
                src={preview}
              />
            ) : (
              "Choose or drop a file."
            )}
            <canvas ref={canvasRef} id="canvas" width="640" height="640" />
           
          </MagicDropzone>
        ) : (
          <div className="Dropzone">Loading model...</div>
          
        )*/}
      </div>
    );
  
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
