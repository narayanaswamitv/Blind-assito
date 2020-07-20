/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.Calendar;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  public static SharedPreferences.Editor editor;
  public static SharedPreferences prefs;

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;
  public static TextToSpeech textToSpeech;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    editor=getSharedPreferences(getResources().getString(R.string.app_name), MODE_PRIVATE).edit();
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    prefs = getSharedPreferences(getResources().getString(R.string.app_name), MODE_PRIVATE);
    String object_name = prefs.getString("object_name", " ");//"No name defined" is the default value.
    Long date_time = prefs.getLong("date_time",0); //0 is the default value.
    Log.d("testing","testing 00001 "+date_time);
    textToSpeech=new TextToSpeech(DetectorActivity.this, new TextToSpeech.OnInitListener() {

      @Override
      public void onInit(int i) {
        if(i == TextToSpeech.SUCCESS){
          int result=textToSpeech.setLanguage(Locale.ENGLISH);
          textToSpeech.isLanguageAvailable(Locale.ENGLISH);
          Log.d("errror","errror 0001 "+result);
          if(result==TextToSpeech.LANG_MISSING_DATA ||
                  result==TextToSpeech.LANG_NOT_SUPPORTED){
            Intent installIntent = new Intent();
            installIntent.setAction(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA);
            startActivity(installIntent);
            Log.e("error", "This Language is not supported");
          }
          else{
            if(date_time==0)
            {
              Log.d("test","testing 01 ");
              Calendar date = Calendar.getInstance();
              long t= date.getTimeInMillis();
              editor.putLong("date_time", t);
              editor.apply();
              if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                Log.d("test","testing 02 ");
                textToSpeech.speak(object_name,TextToSpeech.QUEUE_ADD,null,null);
              } else {
                Log.d("test","testing 03 ");
                textToSpeech.speak(object_name, TextToSpeech.QUEUE_ADD, null);
              }
            }
            ConvertTextToSpeech();
          }
        }
        else
          Log.e("error", "Initilization Failed!");
      }
    });
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            Log.d("testing","testing 00001 ");
            ConvertTextToSpeech();
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }
  private void ConvertTextToSpeech() {
    // TODO Auto-generated method stub
    prefs = getSharedPreferences(getResources().getString(R.string.app_name), MODE_PRIVATE);
    String object_name = prefs.getString("object_name", " ");//"No name defined" is the default value.
    Long date_time = prefs.getLong("date_time",0); //0 is the default value.
    Log.d("checkkkkk","checkkkkk 0006 "+object_name);
    if(object_name==null||"".equals(object_name))
    {
      Log.d("checkkkkk","checkkkkk 0001 ");
      BorderedText.string = "Content not available";
      if(!object_name.equalsIgnoreCase("")) {
        Log.d("checkkkkk","checkkkkk 0002 ");
      if(CheckTime(date_time)) {
          Log.d("checkkkkk","checkkkkk 0003 ");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
          textToSpeech.speak(object_name,TextToSpeech.QUEUE_ADD,null,null);
        } else {
          textToSpeech.speak(object_name, TextToSpeech.QUEUE_ADD, null);
        }
       }
      }
    }else {
      Log.d("checkkkkk","checkkkkk 0005 ");
     if(CheckTime(date_time)) {
        Log.d("checkkkkk","checkkkkk 0004 ");
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
        textToSpeech.speak(object_name,TextToSpeech.QUEUE_ADD,null,null);
        //textToSpeech.setSpeechRate(.5)
      } else {
        textToSpeech.speak(object_name, TextToSpeech.QUEUE_ADD, null);
      }     }
    }
  }
private boolean CheckTime(Long time)
{
  Calendar date = Calendar.getInstance();
  long t= date.getTimeInMillis();
  final long ONE_MINUTE_IN_MILLIS=60000;
  Date afterAddingTenMins=new Date(t + (time * ONE_MINUTE_IN_MILLIS));
  Date date1 = new Date(t);
  Calendar now = Calendar.getInstance();

  now.setTimeInMillis(time);
  now.add(Calendar.SECOND, 3);
  Calendar nowPlus70Minutes = now;
  Date date2=new Date(nowPlus70Minutes.getTimeInMillis());
  Log.d("CheckTime","CheckTime "+date1.toString()+" 011 "+date2);
  if(date1.after(date2))
  {
     date = Calendar.getInstance();
     t= date.getTimeInMillis();
    editor.putLong("date_time", t);
    editor.apply();
    return true;
  }
  else {
    return false;
  }
}
  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
