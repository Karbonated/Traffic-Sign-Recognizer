import sys, cv2, torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

LABELS = {
    0:"Speed limit 20km/h", 1:"Speed limit 30km/h", 2:"Speed limit 50km/h",
    3:"Speed limit 60km/h", 4:"Speed limit 70km/h", 5:"Speed limit 80km/h",
    6:"End of speed limit 80km/h", 7:"Speed limit 100km/h", 8:"Speed limit 120km/h",
    9:"No passing", 10:"No passing for vehicles >3.5t",
    11:"Right-of-way at next intersection", 12:"Priority road",
    13:"Yield", 14:"Stop", 15:"No vehicles",
    16:"Vehicles >3.5t prohibited", 17:"No entry", 18:"General caution",
    19:"Dangerous curve left", 20:"Dangerous curve right",
    21:"Double curve", 22:"Bumpy road", 23:"Slippery road",
    24:"Road narrows on right", 25:"Road works", 26:"Traffic signals",
    27:"Pedestrians", 28:"Children crossing", 29:"Bicycles crossing",
    30:"Beware of ice/snow", 31:"Wild animals crossing",
    32:"End all speed/passing limits", 33:"Turn right ahead",
    34:"Turn left ahead", 35:"Ahead only", 36:"Go straight or right",
    37:"Go straight or left", 38:"Keep right", 39:"Keep left",
    40:"Roundabout mandatory", 41:"End of no passing",
    42:"End no passing for >3.5t"
}

EXPLAIN = {
    14: "Come to a complete stop. Check all directions before proceeding.",
    13: "Yield to oncoming traffic. Slow down and give way.",
    17: "Do not enter. Wrong way.",
    1:  "Maximum speed 30 km/h in this zone.",
    2:  "Maximum speed 50 km/h.",
    38: "Stay to the right side of the road or obstacle.",
    12: "You have priority over other traffic.",
    25: "Road works ahead. Reduce speed and be alert.",
    28: "Watch for children crossing the road.",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf = models.mobilenet_v2(weights=None)
clf.classifier[1] = torch.nn.Linear(clf.last_channel, 43)
clf.load_state_dict(torch.load('gtsrb_mobilenet.pth', map_location=device))
clf = clf.to(device).eval()

tfm = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


yolo = YOLO('yolov8n.pt')   

def classify(crop_bgr):
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    t = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(clf(t), dim=1)
        conf, cls = prob.max(1)
    return cls.item(), conf.item()

def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot open image:", image_path); return

    results = yolo(img, classes=[11], conf=0.1)[0]  
    boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else []

    if len(boxes) == 0:
        print("No signs detected by YOLO. Classifying full image as fallback.")
        cls, conf = classify(img)
        name = LABELS.get(cls, "Unknown")
        exp  = EXPLAIN.get(cls, "Follow standard traffic rules.")
        print(f"\n>>> {name} (confidence: {conf*100:.1f}%)")
        print(f"    {exp}")
        cv2.putText(img, f"{name} {conf*100:.0f}%", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 2)
    else:
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            cls, conf = classify(crop)
            name = LABELS.get(cls, "Unknown")
            exp  = EXPLAIN.get(cls, "Follow standard traffic rules.")
            print(f"\n>>> {name} (confidence: {conf*100:.1f}%)")
            print(f"    {exp}")
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(img, f"{name} {conf*100:.0f}%", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    out = image_path.replace('.', '_result.')
    cv2.imwrite(out, img)
    print(f"\nResult saved to: {out}")

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv)>1 else 'test.jpg')