# -*- coding: utf-8 -*-

import os

import arcpy


class Toolbox(object):
    def __init__(self):
        self.label = "TrainingDataToolbox"
        self.alias = "TrainingDataToolbox"
        self.tools = [ETDTool, SumDataTool]


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
                    row.append(fc_value[field_name])
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
                arcpy.SetProgressorLabel(orig_fc)
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
                l_stats_map[orig_fc] = l_stats
        parameters[0].value = self.create_table(table_name, g_stats, l_stats_map)
        arcpy.ResetProgressor()