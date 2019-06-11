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
        shp_path = arcpy.Parameter(
            displayName="Shapefile Folder",
            name="in_shp_base",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        shp_path.value = os.path.join("G:", os.sep, "DLDATA")

        tif_path = arcpy.Parameter(
            displayName="TIFF Folder",
            name="in_tif_base",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        tif_path.value = os.path.join("G:", os.sep, "DLDATA")

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
        output.value = os.path.join("G:", os.sep, "DLDATA")

        return [shp_path, tif_path, size1, size2, label, output]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, _):
        shp_path = parameters[0].valueAsText
        tif_path = parameters[1].valueAsText
        size1 = parameters[2].value
        size2 = parameters[3].value
        label = parameters[4].value
        output_base = parameters[5].valueAsText

        if not os.path.exists(output_base):
            os.makedirs(output_base)

        label_format = {"VOC": "PASCAL_VOC_rectangles", "RCNN": "RCNN_Masks"}[label]

        arcpy.env.autoCancelling = False
        shp_files = glob.glob(os.path.join(shp_path, "*", "*", "*.shp"))
        for shp_file in shp_files:
            tokens = shp_file.split(os.sep)
            shp_name = tokens[-1]
            area_name = shp_name.split(".")[0]
            tif_name = tokens[-3]
            tif_file = os.path.join(tif_path, tif_name) + ".tif"
            if not os.path.exists(tif_file):
                arcpy.AddWarning("{} does not exist!".format(tif_file))
                break
            output_path = os.path.join(output_base, tif_name, area_name)
            if not os.path.exists(output_path):
                arcpy.ia.ExportTrainingDataForDeepLearning(tif_file,
                                                           output_path,
                                                           shp_file,
                                                           "TIFF",
                                                           size1, size1,
                                                           size2, size2,
                                                           "ONLY_TILES_WITH_FEATURES",
                                                           label_format,
                                                           0,
                                                           "Classvalue",
                                                           0)
            else:
                arcpy.AddWarning("{} already exists.".format(output_path))
            if arcpy.env.isCancelled:
                break
        arcpy.ResetProgressor()
