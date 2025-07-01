import cv2
from inference_sdk import InferenceHTTPClient
import time

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="9k6SBo0cQlOkFConFfVR"
)

def get_inference_prediction(client,image_path:str, threshold = 0.5):
    # Initialize client
    

    # Run workflow
    result = client.run_workflow(
        workspace_name="anthony-intj3",
        workflow_id="custom-workflow-3",
        images={"image": image_path},
        use_cache=True
    )

    # Plot points
    predictions = result[0]['predictions']['predictions']


    points = []

    for pred in predictions:
        # Adjust according to what the response contains
        #print(pred)
        if pred['confidence']<threshold:
            continue
        x, y = int(pred['x']), int(pred['y'])  # center of bbox
        w, h = int(pred['width']), int(pred['height'])
        label = pred.get("class", "")

        # Calculate top-left and bottom-right corners of bounding box
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2
        points.append([x1, y1,x2, y2, label])

    return points


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    img_fname = "temp.jpg"
    time_lapse = 5 #change this to desired time in seconds
    last_saved = time.time()
    data = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        frame = cv2.resize(frame, (640, 360))
        # Save every 5 seconds
        current_time = time.time()
        if current_time - last_saved >= time_lapse:
            cv2.imwrite(img_fname, frame)
            print(f"Saved {img_fname} at {time.strftime('%H:%M:%S')}")
            last_saved = current_time
            data = get_inference_prediction(client,img_fname)
        
        if data is not None:
            for x1,y1,x2,y2,label in data:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1)


        cv2.imshow("Detections", frame)
        cv2.waitKey(1)
