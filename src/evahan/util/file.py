import base64
import re

from PIL import Image
from io import BytesIO

def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_base64 = f"data:image/jpeg;base64,{image_base64}"
    return image_base64

def parser_html_to_json_v2(html_str: str, width: int, height: int, src_width: int, src_height: int) -> dict:
    """
    input html: 
    <div class="text" data-bbox="435 28 479 206">六經圖卷九</div>
    <div class="text" data-bbox="113 0 145 36">目</div>
    <div class="text" data-bbox="230 -1 267 48">黄</div>
    <div class="text" data-bbox="319 -1 359 157">器用制圃</div>
    <div class="book_edge" data-bbox="0 -2 48 756"></div>
    """
    item_list = [itm for itm in html_str.split("\n") if itm.strip() != ""]
    results = []
    for item in item_list:
        try:
            data_bbox_start_indx = item.find("data-bbox=")
            data_bbox_end_indx = item.find("</div>", data_bbox_start_indx)
            bbox = item[data_bbox_start_indx:data_bbox_end_indx]
            bbox = bbox.replace("data-bbox=", "").replace("</div>", "").strip()
            bbox = bbox.split(">")[0]
            bbox = bbox.replace("(", "").replace(")", "")
            bbox = bbox.replace("'", "")
            bbox = bbox.replace('"', "")
            bbox = bbox.replace(">", "")
            print("bbox: ", bbox)
            x1, y1, x2, y2 = [int(x.strip()) for x in bbox.split(" ")]

            cls_name = re.findall(r'class="([^"]+)"', item)
            cls_name = cls_name[0] if len(cls_name) > 0 else ""
            text = re.findall(r'>([^<]+)<', item)
            text = text[0] if len(text) > 0 else ""
            
            x1 = int(x1 / width * src_width)
            y1 = int(y1 / height * src_height)
            x2 = int(x2 / width * src_width)
            y2 = int(y2 / height * src_height)
            bbox = [x1, y1, x2, y2]
            points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ]
            results.append({
                "label": cls_name,
                "text": text,
                "bbox": bbox,
                "points": points
            })
        except Exception as e:
            print(f"error: {e}")
            continue

    return results

def parser_html_to_json(html_str: str, width: int, height: int, src_width: int, src_height: int) -> dict:
    """
    input html: 
    <div class="text" data-bbox="[435, 28, 479, 206]">六經圖卷九</div>
    <div class="text" data-bbox="[113, 0, 145, 36]">目</div>
    <div class="text" data-bbox="[230, -1, 267, 48]">黄</div>
    <div class="text" data-bbox="[319,-1 ,359 ,157]">器用制圃</div>
    <div class="book_edge" data-bbox="[0, -2，48，756]</div>
    """
    item_list = [itm for itm in html_str.split("\n") if itm.strip() != ""]
    results = []
    for item in item_list:
        try:
            bbox = re.findall(r'data-bbox="([^"]+)"', item)
            cls_name = re.findall(r'class="([^"]+)"', item)
            bbox = bbox[0] if len(bbox) > 0 else '[]'
            cls_name = cls_name[0] if len(cls_name) > 0 else ""
            text = re.findall(r'>([^<]+)<', item)
            text = text[0] if len(text) > 0 else ""
            ## bbox '[x1, y1, x2, y2]' to [x1, y1, x2, y2]
            bbox = [int(x.strip()) for x in bbox.strip("[]").split(",") if x.strip() != ""]
            ## each element must > 0 (=0 if not valid)
            bbox = [x if x > 0 else 0 for x in bbox]
            bbox = [x if x < width else width for x in bbox]
            bbox = [y if y < height else height for y in bbox]
            bbox = [y if y > 0 else 0 for y in bbox]
            x1, y1, x2, y2 = bbox
            ## scale the bbox to the size of the image according to the scale factor
            scale_factor_w = width / src_width
            scale_factor_h = height / src_height
            x1 = x1 / scale_factor_w
            y1 = y1 / scale_factor_h
            x2 = x2 / scale_factor_w
            y2 = y2 / scale_factor_h
            bbox = [x1, y1, x2, y2]
            points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ]
            results.append({
                "label": cls_name,
                "text": text,
                "bbox": bbox,
                "points": points,
            })
        except Exception as e:
            print(f"error: {e}")
            continue

    return results