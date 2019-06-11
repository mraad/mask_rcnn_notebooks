# -*- coding: utf-8 -*-

import glob
import os

import arcpy


class Toolbox(object):
    def __init__(self):
        self.label = "ExportTrainingToolbox"
        self.alias = "ExportTrainingToolbox"
        self.tools = [ETDTool]


class ETDTool(object):
    def __init__(self):
        self.label = "Export Training Tool"
        self.description = "Export Training Tool"
        self.canRunInBackground = False

    def getParameterInfo(self):
        shp_base = arcpy.Parameter(
            displayName="Shapefile Base Folder",
            name="in_shp_base",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        shp_base.value = os.path.join("G:", os.sep, "DLDATA")

        tif_base = arcpy.Parameter(
            displayName="TIFF Folder",
            name="in_tif_base",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        tif_base.value = os.path.join("G:", os.sep, "DLDATA")

        mode = arcpy.Parameter(
            displayName="Mode",
            name="in_mode",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        mode.filter.type = "ValueList"
        mode.filter.list = ["Train", "Valid"]
        mode.value = mode.filter.list[0]

        size1 = arcpy.Parameter(
            displayName="Swatch Size",
            name="in_size_1",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        size1.value = 256

        size2 = arcpy.Parameter(
            displayName="Swatch Stride",
            name="in_size_2",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        size2.value = 128

        index = arcpy.Parameter(
            displayName="Starting Index",
            name="in_index",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        index.value = 100000

        label = arcpy.Parameter(
            displayName="Label Format",
            name="in_label",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        label.filter.type = "ValueList"
        label.filter.list = ["RCNN", "VOC"]
        label.value = label.filter.list[0]

        output = arcpy.Parameter(
            displayName="Output Folder",
            name="in_output",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        output.value = os.path.join("G:", os.sep)

        return [shp_base, tif_base, mode, size1, size2, index, label, output]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, _):
        shp_base = parameters[0].valueAsText
        tif_base = parameters[1].valueAsText
        mode = parameters[2].value
        size1 = parameters[3].value
        size2 = parameters[4].value
        index = parameters[5].value
        label = parameters[6].value
        output_base = parameters[7].valueAsText

        if not os.path.exists(output_base):
            os.makedirs(output_base)

        label_format = {"VOC": "PASCAL_VOC_rectangles", "RCNN": "RCNN_Masks"}[label]

        arcpy.env.autoCancelling = False
        shp_files = glob.glob(os.path.join(shp_base, "*", "*", "*.shp"))
        for shp_file in shp_files:
            tokens = shp_file.split(os.sep)
            shp_name = tokens[-1]
            area_name = shp_name.split(".")[0]
            tif_file = os.path.join(tif_base, tokens[-3]) + ".tiff"
            if not os.path.exists(tif_file):
                arcpy.AddWarning("{} does not exist!".format(tif_file))
                break
            output_path = os.path.join(output_base, f"{label}{size1}{mode}", area_name)
            if not os.path.exists(output_path):
                arcpy.ia.ExportTrainingDataForDeepLearning(tif_file,
                                                           output_path,
                                                           shp_file,
                                                           "TIFF",
                                                           size1, size1,
                                                           size2, size2,
                                                           "ONLY_TILES_WITH_FEATURES",
                                                           label_format,
                                                           index,
                                                           "Classvalue",
                                                           0)
            else:
                arcpy.AddWarning("{} already exists.".format(output_path))
            index += 1000
            if arcpy.env.isCancelled:
                break
        arcpy.ResetProgressor()
