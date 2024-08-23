# -*- coding: utf-8 -*-
import cv2
import numpy as np
import requests
from typing import Dict
from requests.adapters import HTTPAdapter

class Decode_Img():

    def __init__(self) -> None:
        self.methods = {"base64": self.base64_to_opencv,
                        "url": self.url_to_opencv,
                        "imgfile": self.buffer_to_opencv,
                        "localfile": self.local_to_opencv}

    def __call__(self, args: Dict):
        for key, method in self.methods.items():
            if args.get(key):
                data = args.get(key)
                img, url, msgs = method(data)
                return img, url, msgs
        return None, None, "decode Img error"

    def url_to_opencv(self, imgurl):
        try:
            session = requests.Session()
            session.mount('http://', HTTPAdapter(max_retries=2))
            session.mount('https://', HTTPAdapter(max_retries=2))

            req = session.get(imgurl, stream=True, timeout=1)
            if req.status_code == 200:
                content = req.content
                nparr = np.frombuffer(content, dtype=np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                msgs = "success"
            else:
                img = None
                msgs = f"img is None, req.status_code:{req.status_code} please check imgurl"

            session.close()

        except Exception as e:
            img = None
            msgs = f"img is None, {e} please check imgurl"
        return img, None, msgs

    def base64_to_opencv(self, base64_code):
        # base64 decode
        import base64
        try:
            img_data = base64.b64decode(base64_code)
            img_array = np.frombuffer(img_data, np.uint8)  # convert into numpy
            img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)  # convert into cv image
            msgs = "success"
        except Exception as e:
            img = None
            msgs = f"img is None, {e} please check base64"
        return img, None, msgs

    def buffer_to_opencv(self, imgfile):
        # imgfile
        from io import BytesIO
        try:
            img = BytesIO(imgfile[0].stream)
            img_array = np.frombuffer(img.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            url = imgfile.filename.split("/")[-1]
            msgs = "success"
        except Exception as e:
            img = url = None
            msgs = f"img is None, {e} please check imgfile"
        return img, url, msgs

    def local_to_opencv(self, localfile):

        img = cv2.imread(str(localfile))
        if isinstance(img, np.ndarray):
            msgs = "success"
        else:
            msgs = f"img is None, please check localfile"
        return img, None, msgs