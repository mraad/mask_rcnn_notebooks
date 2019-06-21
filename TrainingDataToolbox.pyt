# -*- coding: utf-8 -*-

import importlib.util
import os
import re

import arcpy

try:
    import rtree
except:
    pass


class Toolbox(object):
    def __init__(self):
        self.label = "TrainingDataToolbox"
        self.alias = "TrainingDataToolbox"
        self.tools = [ETDTool, SumDataTool, UniqueTool]


class ETDTool(object):
    def __init__(self):
        self.label = "Export Training Data"
        self.description = "Export Training Data"
        self.canRunInBackground = False

    def getParameterInfo(self):
        workspace = arcpy.Parameter(
            displayName="Input workspace",
            name="in_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")
        workspace.value = os.path.join("C", os.sep, "xxxx.sde")

        wild_card = arcpy.Parameter(
            displayName="Feature Class Wild Card",
            name="in_prefix",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        wild_card.value = "*Y*2*"

        class_name = arcpy.Parameter(
            displayName="Class Name",
            name="in_class_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        class_name.filter.type = "ValueList"
        class_name.filter.list = ["Clazz"]
        class_name.value = class_name.filter.list[0]

        class_value = arcpy.Parameter(
            displayName="Class Value",
            name="in_class_value",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        class_value.value = 1

        location = arcpy.Parameter(
            displayName="Location",
            name="in_location",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        location.filter.type = "ValueList"
        location.filter.list = ["Y", "P", "K"]
        location.value = location.filter.list[0]

        mask = arcpy.Parameter(
            displayName="Mask",
            name="in_mask",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        mask.filter.type = "ValueList"
        mask.filter.list = ["Clazz"]
        mask.value = mask.filter.list[0]

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
        size1.value = 300

        size2 = arcpy.Parameter(
            displayName="Swatch Stride",
            name="in_size_2",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        size2.value = 100

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
        label.filter.list = ["VOC", "RCNN"]
        label.value = label.filter.list[0]

        output = arcpy.Parameter(
            displayName="Output Folder",
            name="in_output",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        output.value = os.path.join("C:", os.sep)

        return [workspace, wild_card, class_name, class_value, location, mask, mode, size1, size2, index, label, output]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def create_temp_fc(self, sp_ref):
        ws = "memory"  # arcpy.env.scratchGDB
        fc_name = "TEMP_FC"
        temp_fc = os.path.join(ws, fc_name)
        if arcpy.Exists(temp_fc):
            arcpy.management.Delete(temp_fc)
        arcpy.management.CreateFeatureclass(
            ws,
            fc_name,
            "POLYGON",
            spatial_reference=sp_ref,
            has_m="DISABLED",
            has_z="DISABLED")
        arcpy.management.AddField(temp_fc, "Classname", "TEXT", field_length=128)
        arcpy.management.AddField(temp_fc, "Classvalue", "LONG")
        return temp_fc

    def execute(self, parameters, _):
        workspace = parameters[0].valueAsText
        wild_card = parameters[1].value
        class_name = parameters[2].value
        class_value = parameters[3].value
        location = parameters[4].value
        mask = parameters[5].value
        mode = parameters[6].value
        size1 = parameters[7].value
        size2 = parameters[8].value
        index = parameters[9].value
        label = parameters[10].value
        output_base = parameters[11].valueAsText

        if not os.path.exists(output_base):
            os.makedirs(output_base)

        label_format = {"VOC": "PASCAL_VOC_rectangles", "RCNN": "RCNN_Masks"}[label]
        location_len = len(location)
        sp_ref = arcpy.SpatialReference(4326)

        arcpy.env.autoCancelling = False
        arcpy.env.workspace = workspace
        feature_datasets = arcpy.ListDatasets("*", "All")
        for feature_dataset in feature_datasets:
            for orig_fc in arcpy.ListFeatureClasses(wild_card=wild_card, feature_dataset=feature_dataset):
                arcpy.SetProgressorLabel(orig_fc)

                _, _, last = orig_fc.split(".")
                tif_file = last[location_len:]
                tif_file = os.path.join(output_base, os.sep, "Planet_{}".format(location), "{}.tif".format(tif_file))
                if not os.path.exists(tif_file):
                    arcpy.AddWarning("{} does not exist!".format(tif_file))
                    break
                temp_fc = self.create_temp_fc(sp_ref)
                with arcpy.da.InsertCursor(temp_fc, ["Classname", "Classvalue", "SHAPE@"]) as i_cursor:
                    where = "Classname LIKE '%{}%'".format(class_name)
                    with arcpy.da.SearchCursor(orig_fc, ["SHAPE@"],
                                               where_clause=where,
                                               spatial_reference=sp_ref) as s_cursor:
                        for row in s_cursor:
                            i_cursor.insertRow([class_name, class_value, row[0]])
                output_path = os.path.join(output_base, os.sep, "{}{}{}{}".format(label, mask, size1, mode), last)
                if not os.path.exists(output_path):
                    arcpy.ia.ExportTrainingDataForDeepLearning(tif_file,
                                                               output_path,
                                                               temp_fc,
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
            if arcpy.env.isCancelled:
                break
        arcpy.ResetProgressor()


class SumDataTool(object):
    def __init__(self):
        self.label = "Summarize Data"
        self.description = "Summarize Data"
        self.canRunInBackground = False

    def getParameterInfo(self):
        # https://pro.arcgis.com/en/pro-app/arcpy/classes/parameter.htm

        table = arcpy.Parameter(
            name="out_table",
            displayName="out_table",
            direction="Output",
            datatype="DETable",
            parameterType="Derived")

        workspace = arcpy.Parameter(
            displayName="Input workspace",
            name="in_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")
        workspace.value = os.path.join("E:", os.sep, "ImageClass_zscusw0n121m004.sde")

        wild_card = arcpy.Parameter(
            displayName="Feature Class Wild Card",
            name="in_prefix",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        wild_card.value = "*2017*"

        class_name = arcpy.Parameter(
            displayName="Class Name",
            name="in_class_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        class_name.filter.type = "ValueList"
        class_name.filter.list = ["All"]
        class_name.value = class_name.filter.list[0]

        table_name = arcpy.Parameter(
            displayName="Table Name",
            name="in_table_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        table_name.value = "ObjectStats"

        return [table, workspace, wild_card, class_name, table_name]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def create_table(self, table_name, g_stats, l_stats):
        # ws = "memory"
        ws = arcpy.env.workspace
        temp_table = os.path.join(ws, table_name)
        if arcpy.Exists(temp_table):
            arcpy.management.Delete(temp_table)
        arcpy.management.CreateTable(ws, table_name)
        field_names = ["FeatureClass"]
        arcpy.management.AddField(temp_table, field_names[0], "TEXT", field_length=256)
        for field_name in g_stats:
            field_name = field_name.replace(" ", "")
            arcpy.management.AddField(temp_table, field_name, "LONG")
            field_names.append(field_name)
        with arcpy.da.InsertCursor(temp_table, field_names) as cursor:
            row = ["All"]
            for field_name in g_stats:
                row.append(g_stats[field_name])
            cursor.insertRow(row)
            for fc_name, fc_value in l_stats.items():
                row = [fc_name]
                for field_name in g_stats:
                    row.append(fc_value.get(field_name, 0))
                cursor.insertRow(row)
        return table_name

    def execute(self, parameters, _):
        workspace = parameters[1].valueAsText
        wild_card = parameters[2].value
        class_name = parameters[3].value
        table_name = parameters[4].value

        sp_ref = arcpy.SpatialReference(4326)

        g_stats = {}  # Key is class name, value is the global count over all the feature classes.
        l_stats_map = {}  # Key is feature class name, value is a map of local class name -> count.
        arcpy.env.autoCancelling = False
        arcpy.env.workspace = workspace
        feature_datasets = arcpy.ListDatasets("*", "All")
        for feature_dataset in feature_datasets:
            if arcpy.env.isCancelled:
                break
            for orig_fc in arcpy.ListFeatureClasses(wild_card=wild_card, feature_dataset=feature_dataset):
                if arcpy.env.isCancelled:
                    break
                key = orig_fc.split(".")[-1]
                arcpy.SetProgressorLabel(key)
                if class_name == "All":
                    where = "1=1"
                else:
                    where = f"Classname LIKE '%{class_name}%'"
                l_stats = {}
                with arcpy.da.SearchCursor(orig_fc,
                                           ["Classname"],
                                           where_clause=where,
                                           spatial_reference=sp_ref) as s_cursor:
                    for row in s_cursor:
                        row_class_name = row[0]
                        l_stats.setdefault(row_class_name, 0)
                        l_stats[row_class_name] += 1
                        g_stats.setdefault(row_class_name, 0)
                        g_stats[row_class_name] += 1
                l_stats_map[key] = l_stats
        parameters[0].value = self.create_table(table_name, g_stats, l_stats_map)
        arcpy.ResetProgressor()


class ObjectCount(object):
    def __init__(self, geom, name, facility):
        self.geom = geom
        self.name = name
        self.facility = facility
        self.count = 1

    def iou(self, geom):
        if self.geom.disjoint(geom):
            return 0.0
        inter = self.geom.intersect(geom).area
        union = self.geom.union(geom).area
        return inter / union

    def is_same(self, facility, name, geom):
        return self.facility == facility and self.name == name and self.iou(geom) > 0.5

    def increment_count(self):
        self.count += 1

    def to_row(self):
        return self.facility, self.name, self.count, self.geom


class UniqueTool(object):
    def __init__(self):
        self.label = "Unique Count"
        self.description = "Unique Count"
        self.canRunInBackground = False

    def getParameterInfo(self):
        # https://pro.arcgis.com/en/pro-app/arcpy/classes/parameter.htm

        feature_layer = arcpy.Parameter(
            name="out_fl",
            displayName="out_fl",
            direction="Output",
            datatype="Feature Layer",
            parameterType="Derived")

        workspace = arcpy.Parameter(
            displayName="Input workspace",
            name="in_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")
        workspace.value = os.path.join("E:", os.sep, "ImageClass_zscusw0n121m004.sde")

        wild_card = arcpy.Parameter(
            displayName="Feature Class Wild Card",
            name="in_prefix",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        wild_card.value = "*2017*"

        class_name = arcpy.Parameter(
            displayName="Class Name",
            name="in_class_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        class_name.filter.type = "ValueList"
        class_name.filter.list = ["All"]
        class_name.value = class_name.filter.list[0]

        layer_name = arcpy.Parameter(
            displayName="Layer Name",
            name="in_layer_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        layer_name.value = "ObjectStats"

        return [feature_layer, workspace, wild_card, class_name, layer_name]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def create_feature_class(self, layer_name, arr, sp_ref):
        # ws = "memory"
        ws = arcpy.env.workspace
        feature_class = os.path.join(ws, layer_name)
        if arcpy.Exists(feature_class):
            arcpy.management.Delete(feature_class)
        arcpy.management.CreateFeatureclass(ws,
                                            layer_name,
                                            "POLYGON",
                                            spatial_reference=sp_ref,
                                            has_m="DISABLED",
                                            has_z="DISABLED")
        field_names = ["Facility", "Classname", "Population", "SHAPE@"]
        arcpy.management.AddField(feature_class, field_names[0], "TEXT", field_length=128)
        arcpy.management.AddField(feature_class, field_names[1], "TEXT", field_length=128)
        arcpy.management.AddField(feature_class, field_names[2], "LONG")
        with arcpy.da.InsertCursor(feature_class, field_names) as cursor:
            for o in arr:
                cursor.insertRow(o.to_row())
        return feature_class

    def execute(self, parameters, _):
        spec = importlib.util.find_spec("rtree")
        if spec is None:
            arcpy.AddError("Please install 'rtree' package using the Python Package Manager")
            return
        workspace = parameters[1].valueAsText
        wild_card = parameters[2].value
        class_name = parameters[3].value
        layer_name = parameters[4].value

        sp_ref = arcpy.SpatialReference(4326)
        pattern = re.compile("(\w+)\d.+")
        oid = 0
        arr = []
        sp_index = rtree.index.Index()

        arcpy.env.autoCancelling = False
        arcpy.env.workspace = workspace
        feature_datasets = arcpy.ListDatasets("*", "All")
        for feature_dataset in feature_datasets:
            if arcpy.env.isCancelled:
                break
            for fc in arcpy.ListFeatureClasses(wild_card=wild_card, feature_dataset=feature_dataset):
                if arcpy.env.isCancelled:
                    break
                fc_name = fc.split(".")[-1]
                match = pattern.match(fc_name)
                facility = match.group(1)
                arcpy.SetProgressorLabel(fc_name)
                if class_name == "All":
                    where = "1=1"
                else:
                    where = f"Classname LIKE '%{class_name}%'"
                with arcpy.da.SearchCursor(fc,
                                           ["Classname", "SHAPE@"],
                                           where_clause=where,
                                           spatial_reference=sp_ref) as cursor:
                    for row in cursor:
                        name = row[0]
                        geom = row[1]
                        extent = geom.extent
                        bounds = (extent.XMin, extent.YMin, extent.XMax, extent.YMax)
                        found = False
                        for elem in sp_index.intersection(bounds, objects=True):
                            if elem.object.is_same(facility, name, geom):
                                elem.object.increment_count()
                                found = True
                                break
                        if not found:
                            object_count = ObjectCount(geom, name, facility)
                            sp_index.insert(oid, bounds, object_count)
                            arr.append(object_count)
                            oid += 1
        parameters[0].value = self.create_feature_class(layer_name, arr, sp_ref)
        arcpy.ResetProgressor()
