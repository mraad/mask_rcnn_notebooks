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
        self.tools = [ETDTool, SumDataTool, UniqueTool, CopyFeaturesTool, CleanClassNameTool]


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
        workspace.value = os.path.join("E:", os.sep, "ImageClass_zscusw0n121m004.sde")

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
        class_name.filter.list = ["E_PipeCompleted", "Smoke stack"]
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
        location.filter.list = ["Yanbu", "Paradip", "Karachi"]
        location.value = location.filter.list[0]

        mask = arcpy.Parameter(
            displayName="Mask",
            name="in_mask",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        mask.filter.type = "ValueList"
        mask.filter.list = ["Pipes", "Smoke"]
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
        label.filter.list = ["VOC", "RCNN"]
        label.value = label.filter.list[1]

        output = arcpy.Parameter(
            displayName="Output Folder",
            name="in_output",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        output.value = os.path.join("E:", os.sep)

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
            if arcpy.env.isCancelled:
                break
            for orig_fc in arcpy.ListFeatureClasses(wild_card=wild_card, feature_dataset=feature_dataset):
                if arcpy.env.isCancelled:
                    break
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
        ws = arcpy.env.scratchGDB
        temp_table = os.path.join(ws, table_name)
        if arcpy.Exists(temp_table):
            arcpy.management.Delete(temp_table)
        arcpy.management.CreateTable(ws, table_name)
        field_names = ["FeatureClass"]
        arcpy.management.AddField(temp_table, field_names[0], "TEXT", field_length=256)
        for field_name in g_stats:
            field_name = field_name.replace(" ", "_").replace("(", "").replace(")", "")
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
                        # https://inventwithpython.com/blog/2019/06/05/pythonic-ways-to-use-dictionaries/
                        l_stats.setdefault(row_class_name, 0)
                        l_stats[row_class_name] += 1
                        g_stats.setdefault(row_class_name, 0)
                        g_stats[row_class_name] += 1
                l_stats_map[key] = l_stats
        # Put the table in the ToC
        parameters[0].value = self.create_table(table_name, g_stats, l_stats_map)
        arcpy.ResetProgressor()


class ObjectCount(object):
    # Supporting class to Unique Tool
    def __init__(self, geom, name, facility, last_fc):
        self.geom = geom
        self.name = name
        self.facility = facility
        self.last_fc = last_fc
        self.count = 1

    def iou(self, geom):
        # https://pro.arcgis.com/en/pro-app/arcpy/classes/geometry.htm
        ret_val = 0.0
        if not self.geom.disjoint(geom):
            inter = self.geom.intersect(geom, 4).area  # 4 is for polygon !
            union = self.geom.union(geom).area
            ret_val = inter / union
        return ret_val

    def is_same(self, facility, name, geom, iou_min):
        return self.facility == facility and self.name == name and self.iou(geom) > iou_min

    def increment_count(self, last_fc):
        self.last_fc = last_fc
        self.count += 1

    def to_row(self):
        return self.facility, self.name, self.count, self.last_fc, self.geom


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

        iou_threshold = arcpy.Parameter(
            displayName="IoU Threshold",
            name="in_iou_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        iou_threshold.value = 0.35

        return [feature_layer, workspace, wild_card, class_name, layer_name, iou_threshold]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def create_feature_class(self, layer_name, arr, sp_ref):
        # ws = "memory"
        ws = arcpy.env.scratchGDB
        feature_class = os.path.join(ws, layer_name)
        if arcpy.Exists(feature_class):
            arcpy.management.Delete(feature_class)
        arcpy.management.CreateFeatureclass(ws,
                                            layer_name,
                                            "POLYGON",
                                            spatial_reference=sp_ref,
                                            has_m="DISABLED",
                                            has_z="DISABLED")
        field_names = ["Facility", "Classname", "Population", "LastFeatureClass", "SHAPE@"]
        arcpy.management.AddField(feature_class, field_names[0], "TEXT", field_length=128)
        arcpy.management.AddField(feature_class, field_names[1], "TEXT", field_length=128)
        arcpy.management.AddField(feature_class, field_names[2], "LONG")
        arcpy.management.AddField(feature_class, field_names[3], "TEXT", field_length=128)
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
        iou_min = parameters[5].value

        sp_ref = arcpy.SpatialReference(4326)
        pattern = re.compile("([^\d]+).+")
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
                        for elem in sp_index.intersection(bounds):
                            object_count = arr[elem]
                            if object_count.is_same(facility, name, geom, iou_min):
                                object_count.increment_count(fc_name)
                                found = True
                                break
                        if not found:
                            sp_index.insert(oid, bounds)
                            arr.append(ObjectCount(geom, name, facility, fc_name))
                            oid += 1
        parameters[0].value = self.create_feature_class(layer_name, arr, sp_ref)
        arcpy.ResetProgressor()


class CopyFeaturesTool(object):
    def __init__(self):
        self.label = "Copy Features"
        self.description = "Copy Features"
        self.canRunInBackground = False

    def getParameterInfo(self):
        # https://pro.arcgis.com/en/pro-app/arcpy/classes/parameter.htm

        from_fc = arcpy.Parameter(
            name="from_fc",
            displayName="From Feature Class",
            direction="Input",
            datatype="Table View",
            parameterType="Required")

        workspace = arcpy.Parameter(
            name="in_workspace",
            displayName="To Workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")
        workspace.value = os.path.join("E:", os.sep, "ImageClass_zscusw0n121m004.sde")

        wild_card = arcpy.Parameter(
            name="in_prefix",
            displayName="To Feature Class Wild Card",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        wild_card.value = "*2017*"

        return [from_fc, workspace, wild_card]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, _):
        from_fc = parameters[0].valueAsText
        workspace = parameters[1].valueAsText
        wild_card = parameters[2].value
        arcpy.env.autoCancelling = False

        # def _yield_rows():
        #     field_names = ["SHAPE", "Classname", "Classvalue"]
        #     with arcpy.da.SearchCursor(from_fc, field_names) as cursor:
        #         for row in cursor:
        #             yield row

        # Make sure that input feature class has the fields Classname and Classvalue.
        found = 0
        description = arcpy.Describe(from_fc)
        for field in description.fields:
            if field.name in ["Classname", "Classvalue"]:
                found += 1
        if found != 2:
            arcpy.AddError("Input feature class does not have Classname and/or Classvalue as fields")
        else:
            arcpy.env.workspace = workspace
            datasets = arcpy.ListDatasets("*", "Feature")
            arcpy.AddMessage(f"Found {len(datasets)} datasets.")
            for dataset in datasets:
                if arcpy.env.isCancelled:
                    break
                arcpy.AddMessage(dataset)
                for dest_fc in arcpy.ListFeatureClasses(wild_card=wild_card,
                                                        feature_type="Polygon",
                                                        feature_dataset=dataset):
                    if arcpy.env.isCancelled:
                        break
                    arcpy.SetProgressorLabel(dest_fc)
                    arcpy.management.Append(from_fc, dest_fc, "NO_TEST")
            # Look for features in the current workspace in case it is a FileGDB.
            for dest_fc in arcpy.ListFeatureClasses(wild_card=wild_card,
                                                    feature_type="Polygon",
                                                    feature_dataset=""):
                if arcpy.env.isCancelled:
                    break
                arcpy.SetProgressorLabel(dest_fc)
                arcpy.management.Append(from_fc, dest_fc, "NO_TEST")
            arcpy.ResetProgressor()


class CleanClassNameTool(object):
    def __init__(self):
        self.label = "Clean Class Names"
        self.description = "Clean Class Names"
        self.canRunInBackground = False

    def getParameterInfo(self):
        # https://pro.arcgis.com/en/pro-app/arcpy/classes/parameter.htm

        workspace = arcpy.Parameter(
            name="in_workspace",
            displayName="To Workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")
        workspace.value = os.path.join("E:", os.sep, "ImageClass_zscusw0n121m004.sde")

        wild_card = arcpy.Parameter(
            name="in_prefix",
            displayName="To Feature Class Wild Card",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        wild_card.value = "*2017*"

        return [workspace, wild_card]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, _):
        workspace = parameters[0].valueAsText
        wild_card = parameters[1].value
        arcpy.env.autoCancelling = False

        def calc_field(fc):
            arcpy.management.CalculateField(fc,
                                            "Classname",
                                            """re.sub('[\s+]', '', !Classname!)""",
                                            "PYTHON3",
                                            "import re")

        arcpy.env.workspace = workspace
        datasets = arcpy.ListDatasets("*", "Feature")
        arcpy.AddMessage(f"Found {len(datasets)} datasets.")
        for dataset in datasets:
            if arcpy.env.isCancelled:
                break
            arcpy.AddMessage(dataset)
            for dest_fc in arcpy.ListFeatureClasses(wild_card=wild_card,
                                                    feature_type="Polygon",
                                                    feature_dataset=dataset):
                if arcpy.env.isCancelled:
                    break
                arcpy.SetProgressorLabel(dest_fc)
                calc_field(dest_fc)
        # Look for features in the current workspace in case it is a FileGDB.
        for dest_fc in arcpy.ListFeatureClasses(wild_card=wild_card,
                                                feature_type="Polygon",
                                                feature_dataset=""):
            if arcpy.env.isCancelled:
                break
            arcpy.SetProgressorLabel(dest_fc)
            calc_field(dest_fc)
        arcpy.ResetProgressor()
