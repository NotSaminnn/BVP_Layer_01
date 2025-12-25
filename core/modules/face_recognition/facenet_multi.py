import os, sys, pickle, argparse, glob

# Fix subprocess errors on Windows - must be done before other imports
import platform
if platform.system() == "Windows":
    import multiprocessing
    multiprocessing.freeze_support()
    # Set start method to 'spawn' to avoid subprocess errors on Windows
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, or cannot be changed
        pass

import numpy as np, cv2, torch
from facenet_pytorch import MTCNN, InceptionResnetV1

GALLERY_PATH = "gallery.pkl"  # {name: {"emb": np(512,), "n": int}}

# ----------------- Core -----------------
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def init_models(dev):
    # Initialize MTCNN with optimized settings for speed while maintaining accuracy
    try:
        mtcnn = MTCNN(
            image_size=160, 
            margin=20, 
            post_process=True, 
            device=dev,
            keep_all=False,
            min_face_size=40,  # Increased for speed (skip very small faces)
            thresholds=[0.7, 0.8, 0.8],  # Higher thresholds = faster, still accurate
            factor=0.709
        )
    except Exception as e:
        print(f"Warning: MTCNN initialization issue: {e}", file=sys.stderr)
        # Fallback with minimal settings
        mtcnn = MTCNN(image_size=160, margin=20, device=dev, post_process=True)
    
    net = InceptionResnetV1(pretrained="vggface2").eval().to(dev)
    return mtcnn, net

def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_for_detection(img, max_size=800):
    """Resize image for faster face detection while maintaining aspect ratio"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

@torch.no_grad()
def embed_faces_bgr(frame_bgr, mtcnn, net, dev):
    try:
        rgb = bgr2rgb(frame_bgr)
        # Use detect() to get boxes, then forward() to get faces
        boxes, probs = mtcnn.detect(rgb)
        if boxes is None:
            return None, None
        
        # Get aligned face images
        faces = mtcnn(rgb)
        if faces is None:
            return None, None
        
        # Handle single face vs multiple faces
        if faces.dim() == 3:
            faces = faces.unsqueeze(0)
        
        emb = net(faces.to(dev)).cpu()
        emb = torch.nn.functional.normalize(emb, dim=1)  # L2-normalized
        return emb.numpy(), boxes
    except (OSError, RuntimeError, AttributeError) as e:
        # Handle subprocess and multiprocessing errors
        if "subprocess" in str(e).lower() or "multiprocessing" in str(e).lower():
            print(f"Error in face detection (subprocess issue): {e}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Error in face detection: {e}", file=sys.stderr)
        return None, None

def load_gallery():
    if os.path.exists(GALLERY_PATH):
        with open(GALLERY_PATH, "rb") as f: return pickle.load(f)
    return {}

def save_gallery(gal): 
    with open(GALLERY_PATH, "wb") as f: pickle.dump(gal, f)

def add_samples(gallery, name, samples):  # samples: [512]
    mean_new = np.mean(np.vstack(samples), axis=0)
    mean_new /= (np.linalg.norm(mean_new) + 1e-9)
    if name in gallery:
        old, n = gallery[name]["emb"], gallery[name]["n"]
        merged = (old * n + mean_new * len(samples)) / (n + len(samples))
        merged /= (np.linalg.norm(merged) + 1e-9)
        gallery[name] = {"emb": merged, "n": n + len(samples)}
    else:
        gallery[name] = {"emb": mean_new, "n": len(samples)}

# ----------------- Commands -----------------
def cmd_enroll(args):
    dev = device()
    print(f"Using: {'GPU (CUDA)' if dev == 'cuda' else 'CPU'}")
    mtcnn, net = init_models(dev)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): print("ERROR: camera open failed", file=sys.stderr); return
    name = args.name.strip()
    collected = []
    print(f"[Enroll] {name}: capturing {args.shots} samples. Press q to stop.")
    
    # Process every Nth frame for speed
    frame_count = 0
    process_interval = 2 if dev == 'cpu' else 1  # Skip frames on CPU
    
    while len(collected) < args.shots:
        ok, frame = cap.read()
        if not ok: break
        
        frame_count += 1
        if frame_count % process_interval == 0:
            # Resize for faster detection
            display_frame = frame.copy()
            small_frame = resize_for_detection(frame, max_size=640)
            embs, boxes = embed_faces_bgr(small_frame, mtcnn, net, dev)
            
            if embs is not None:
                # Scale boxes back to original frame size
                scale = frame.shape[1] / small_frame.shape[1]
                boxes = boxes * scale
                
                # largest face
                areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in boxes ]
                i = int(np.argmax(areas))
                collected.append(embs[i])
                x1,y1,x2,y2 = boxes[i].astype(int)
                cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,255,0),2)
            
            cv2.putText(display_frame, f"{len(collected)}/{args.shots}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("Enroll", display_frame)
        else:
            cv2.imshow("Enroll", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release(); cv2.destroyAllWindows()
    if collected:
        gal = load_gallery(); add_samples(gal, name, collected); save_gallery(gal)
        print(f"Saved {len(collected)} samples to '{name}'. Total images: {gal[name]['n']}")
    else:
        print("No samples saved.")

def cmd_enroll_dir(args):
    import time
    start_time = time.time()
    
    dev = device()
    print(f"Using: {'GPU (CUDA) - Fast' if dev == 'cuda' else 'CPU - Slower'}")
    if dev == 'cpu':
        print("Tip: Install CUDA-enabled PyTorch for 5-10x speedup!")
    
    print("Loading models...", end=" ", flush=True)
    mtcnn, net = init_models(dev)
    print("✓")
    
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        paths.extend(glob.glob(os.path.join(args.dir, ext)))
    paths.sort()
    if not paths: 
        print("No images found."); return
    
    print(f"Processing {len(paths)} images...")
    samples = []
    for idx, p in enumerate(paths, 1):
        if idx % 5 == 0 or idx == 1 or idx == len(paths):
            print(f"  Progress: {idx}/{len(paths)}", flush=True)
        
        img = cv2.imread(p)
        if img is None: continue
        
        # Resize large images for faster processing
        img = resize_for_detection(img, max_size=800)
        
        embs, boxes = embed_faces_bgr(img, mtcnn, net, dev)
        if embs is None: continue
        
        # choose largest face
        areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in boxes ]
        i = int(np.argmax(areas))
        samples.append(embs[i])
    
    elapsed = time.time() - start_time
    
    if not samples:
        print("No faces detected in folder.")
        return
    
    gal = load_gallery(); add_samples(gal, args.name.strip(), samples); save_gallery(gal)
    print(f"✓ Enrolled '{args.name}' from {len(samples)} images in {elapsed:.1f}s ({elapsed/len(samples):.1f}s per image)")

def cmd_list(_):
    gal = load_gallery()
    if not gal: print("Gallery empty."); return
    print("People in gallery:")
    for k,v in gal.items():
        print(f" - {k} (images: {v['n']})")

def cmd_remove(args):
    gal = load_gallery()
    if args.name in gal:
        del gal[args.name]; save_gallery(gal); print(f"Removed '{args.name}'.")
    else:
        print(f"'{args.name}' not found.")

def cmd_clear(_):
    if os.path.exists(GALLERY_PATH): os.remove(GALLERY_PATH)
    print("Cleared gallery.")

def cmd_recognize(args):
    gal = load_gallery()
    if not gal:
        print("Gallery empty. Enroll first."); return
    names = list(gal.keys())
    G = np.stack([gal[n]["emb"] for n in names], axis=0)  # [K,512], L2-normalized
    dev = device()
    print(f"Using: {'GPU (CUDA)' if dev == 'cuda' else 'CPU'}")
    mtcnn, net = init_models(dev)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): print("ERROR: camera open failed", file=sys.stderr); return

    TH = args.threshold
    print(f"[Recognize] People: {', '.join(names)} | threshold={TH}. Press q to quit.")
    
    # Process every Nth frame for real-time performance
    frame_count = 0
    process_interval = 3 if dev == 'cpu' else 1  # Process every 3rd frame on CPU
    last_results = None  # Cache last detection results
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Only process some frames for speed
        if frame_count % process_interval == 0:
            small_frame = resize_for_detection(frame, max_size=640)
            embs, boxes = embed_faces_bgr(small_frame, mtcnn, net, dev)
            
            if embs is not None:
                # Scale boxes back to original frame size
                scale = frame.shape[1] / small_frame.shape[1]
                boxes = boxes * scale
                
                last_results = []
                for e, b in zip(embs, boxes):
                    e = e / (np.linalg.norm(e)+1e-9)
                    sims = G @ e  # cosine sim vs all
                    j = int(np.argmax(sims))
                    best, who = float(sims[j]), names[j]
                    label = who if best >= TH else "Unknown"
                    last_results.append((label, best, b))
            else:
                last_results = None
        
        # Draw cached results
        if last_results:
            for label, best, b in last_results:
                color = (0,255,0) if label!="Unknown" else (0,0,255)
                x1,y1,x2,y2 = b.astype(int)
                cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(display_frame, f"{label} {best:.2f}", (x1, max(0,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif frame_count > process_interval:
            cv2.putText(display_frame,"No face detected",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        
        cv2.imshow("Recognize", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release(); cv2.destroyAllWindows()

# ----------------- CLI -----------------
def main():
    p = argparse.ArgumentParser("FaceNet multi-person recognition (MTCNN + InceptionResnetV1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="Enroll one person from webcam.")
    e.add_argument("--name", required=True)
    e.add_argument("--shots", type=int, default=25)
    e.add_argument("--camera", type=int, default=0)
    e.set_defaults(func=cmd_enroll)

    ed = sub.add_parser("enroll-dir", help="Enroll one person from an image folder.")
    ed.add_argument("--name", required=True)
    ed.add_argument("--dir", required=True)
    ed.set_defaults(func=cmd_enroll_dir)

    r = sub.add_parser("recognize", help="Recognize faces from webcam.")
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--threshold", type=float, default=0.82)
    r.set_defaults(func=cmd_recognize)

    l = sub.add_parser("list", help="List enrolled people.")
    l.set_defaults(func=cmd_list)

    rm = sub.add_parser("remove", help="Remove a person.")
    rm.add_argument("--name", required=True)
    rm.set_defaults(func=cmd_remove)

    c = sub.add_parser("clear", help="Clear the whole gallery.")
    c.set_defaults(func=cmd_clear)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
