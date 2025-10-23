# -- coding: utf-8 --

import sys
from ctypes import *
import cv2
import numpy as np

from OPTSDK.OPTApi import OPTCamera

# 添加OPTSDK路径
sys.path.append("E:\\pyFile\\Multi-Exposure Image Fusion\\OPTSDK")

from OPTSDK.OPTDefines import typeGigeCamera, typeU3vCamera, OPT_OK, OPT_DeviceList, OPT_PixelConvertParam, OPT_Frame, \
    OPT_EInterfaceType, OPT_ECreateHandleMode, OPT_EBayerDemosaic, OPT_EPixelType
from OPTApi import *

# 显示设备信息的函数
def displayDeviceInfo(deviceInfoList):
    print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
    print("------------------------------------------------------------------------------------------------")
    for i in range(0, deviceInfoList.nDevNum):
        pDeviceInfo = deviceInfoList.pDevInfo[i]
        strType = ""
        strVendorName = ""
        strModeName = ""
        strSerialNumber = ""
        strCameraname = ""
        strIpAdress = ""
        for str in pDeviceInfo.vendorName:
            strVendorName = strVendorName + chr(str)
        for str in pDeviceInfo.modelName:
            strModeName = strModeName + chr(str)
        for str in pDeviceInfo.serialNumber:
            strSerialNumber = strSerialNumber + chr(str)
        for str in pDeviceInfo.cameraName:
            strCameraname = strCameraname + chr(str)
        for str in pDeviceInfo.DeviceSpecificInfo.gigeDeviceInfo.ipAddress:
            strIpAdress = strIpAdress + chr(str)
        if pDeviceInfo.nCameraType == typeGigeCamera:
            strType = "Gige"
        elif pDeviceInfo.nCameraType == typeU3vCamera:
            strType = "U3V"
        print("[%d]  %s   %s    %s      %s     %s           %s" % (i + 1, strType, strVendorName, strModeName, strSerialNumber, strCameraname, strIpAdress))

# 让用户选择转换格式的函数
def selectConvertFormat():
    convertFormatCnt = 4
    print("--------------------------------------------")
    print("\t0.Convert to mono8")
    print("\t1.Convert to RGB24")
    print("\t2.Convert to BGR24")
    print("\t3.Convert to BGRA32")
    print("--------------------------------------------")
    inputIndex = input("Please input convert pixelformat: ")

    if int(inputIndex) > convertFormatCnt | int(inputIndex) < 0:
        print("input error!")
        return OPT_INVALID_PARAM
    inputIndex = int(inputIndex)
    if 0 == inputIndex:
        convertFormat = OPT_EPixelType.gvspPixelMono8
    elif 1 == inputIndex:
        convertFormat = OPT_EPixelType.gvspPixelRGB8
    elif 2 == inputIndex:
        convertFormat = OPT_EPixelType.gvspPixelBGR8
    elif 3 == inputIndex:
        convertFormat = OPT_EPixelType.gvspPixelBGRA8
    else:
        convertFormat = OPT_EPixelType.gvspPixelMono8

    return convertFormat

# 定义一个函数，用于从相机读取图像序列
def capture_images_from_camera(cam, frame, num_images):
    images = []
    while len(images) < num_images:
        ret = cam.OPT_GetFrame(frame, 500)
        if ret == OPT_OK:
            image = np.ctypeslib.as_array(frame.pData).reshape((frame.frameInfo.height, frame.frameInfo.width, 3))
            images.append(image)
        else:
            print(f"Failed to capture image {len(images) + 1}")
            break
    return images

# 定义一个函数，用于进行曝光融合
def exposure_fusion(images):
    # 创建一个对齐对象，用于对输入图像进行对齐
    alignMTB = cv2.createAlignMTB()
    # 对图像列表进行对齐处理，结果存储在同一个列表中
    alignMTB.process(images, images)

    # 创建一个合并对象，用于曝光融合
    mergeMertens = cv2.createMergeMertens()
    # 使用合并对象处理图像列表，得到曝光融合的结果
    exposureFusion = mergeMertens.process(images)

    return exposureFusion

# 图像转换的函数
def imageConvert(cam, frame, convertFormat):
    stPixelConvertParam = OPT_PixelConvertParam()

    if OPT_EPixelType.gvspPixelRGB8 == convertFormat:
        nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3
        FileName = "convertRGB8.bin"
        pConvertFormatStr = "RGB8"
    elif OPT_EPixelType.gvspPixelBGR8 == convertFormat:
        nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3
        FileName = "convertBGR8.bin"
        pConvertFormatStr = "BGR8"
    elif OPT_EPixelType.gvspPixelBGRA8 == convertFormat:
        nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 4
        FileName = "convertBGRA8.bin"
        pConvertFormatStr = "BGRA8"
    else:
        nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
        FileName = "convertMono8.bin"
        pConvertFormatStr = "Mono8"

    pDstBuf = (c_ubyte * nDstBufSize)()
    memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))
    stPixelConvertParam.nWidth = frame.frameInfo.width
    stPixelConvertParam.nHeight = frame.frameInfo.height
    stPixelConvertParam.ePixelFormat = frame.frameInfo.pixelFormat
    stPixelConvertParam.pSrcData = frame.pData
    stPixelConvertParam.nSrcDataLen = frame.frameInfo.size
    stPixelConvertParam.nPaddingX = frame.frameInfo.paddingX
    stPixelConvertParam.nPaddingY = frame.frameInfo.paddingY
    stPixelConvertParam.eBayerDemosaic = OPT_EBayerDemosaic.demosaicNearestNeighbor
    stPixelConvertParam.eDstPixelFormat = convertFormat
    stPixelConvertParam.pDstBuf = pDstBuf
    stPixelConvertParam.nDstBufSize = nDstBufSize

    nRet = cam.OPT_PixelConvert(stPixelConvertParam)
    if OPT_OK == nRet:
        print("image convert to %s successfully! nDstDataLen (%d)" % (pConvertFormatStr, stPixelConvertParam.nDstBufSize))
        hFile = open(FileName.encode('ascii'), "wb+")
        try:
            img_buff = (c_ubyte * stPixelConvertParam.nDstBufSize)()
            cdll.msvcrt.memcpy(byref(img_buff), stPixelConvertParam.pDstBuf, stPixelConvertParam.nDstBufSize)
            hFile.write(img_buff)
        except:
            print("save file executed failed")
        finally:
            hFile.close()
    else:
        print("image convert to %s failed! ErrorCode[%d]" % (pConvertFormatStr, nRet))
        del pDstBuf
        sys.exit()

    if pDstBuf is not None:
        del pDstBuf

# 主程序
if __name__ == "__main__":
    deviceList = OPT_DeviceList()
    interfaceType = OPT_EInterfaceType.interfaceTypeAll
    frame = OPT_Frame()
    nRet = OPTCamera.OPT_EnumDevices(deviceList, interfaceType)
    if OPT_OK != nRet:
        print("Enumeration devices failed! ErrorCode", nRet)
        sys.exit()
    if deviceList.nDevNum == 0:
        print("find no device!")
        sys.exit()

    print("deviceList size is", deviceList.nDevNum)
    displayDeviceInfo(deviceList)

    nConnectionNum = input("Please input the camera index: ")

    if int(nConnectionNum) > deviceList.nDevNum:
        print("intput error!")
        sys.exit()

    cam = OPTCamera()
    # 创建设备句柄
    nRet = cam.OPT_CreateHandle(OPT_ECreateHandleMode.modeByIndex, byref(c_void_p(int(nConnectionNum) - 1)))
    if OPT_OK != nRet:
        print("Create devHandle failed! ErrorCode", nRet)
        sys.exit()

    # 打开相机
    nRet = cam.OPT_Open()
    if OPT_OK != nRet:
        print("Open devHandle failed! ErrorCode", nRet)
        sys.exit()

    # 开始拉流
    nRet = cam.OPT_StartGrabbing()
    if OPT_OK != nRet:
        print("Start grabbing failed! ErrorCode", nRet)
        sys.exit()

    # 取一帧图像
    nRet = cam.OPT_GetFrame(frame, 500)
    if OPT_OK != nRet:
        print("Get frame failed!ErrorCode[%d]" % nRet)
        sys.exit()

    # 选择图像转换目标格式
    convertFormat = selectConvertFormat()

    print("BlockId (%d) pixelFormat (%d), Start image convert..." % (
    frame.frameInfo.blockId, frame.frameInfo.pixelFormat))

    # 图片转化
    imageConvert(cam, frame, convertFormat)

    # 释放图像缓存
    nRet = cam.OPT_ReleaseFrame(frame)
    if OPT_OK != nRet:
        print("Release frame failed!Errorcode[%d]" % nRet)
        sys.exit()

    # 停止拉流
    nRet = cam.OPT_StopGrabbing()
    if OPT_OK != nRet:
        print("Stop grabbing failed! ErrorCode", nRet)
        sys.exit()

    # 关闭相机
    nRet = cam.OPT_Close()
    if OPT_OK != nRet:
        print("Close camera failed! ErrorCode", nRet)
        sys.exit()

    # 销毁句柄
    if (cam.handle):
        nRet = cam.OPT_DestroyHandle()

    print("---Demo end---")