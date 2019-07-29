from collections import defaultdict

import cv2
import fiona
from fiona.crs import from_epsg
from keras.layers import Input, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from shapely.geometry import mapping, Polygon, MultiPolygon


def save_polygons(shp_name, geom, wkid=3857):
    schema = {
        'geometry': 'Polygon',
        'properties': {'fid': 'int'},
    }
    with fiona.open(shp_name, 'w', 'ESRI Shapefile', schema, crs=from_epsg(wkid)) as fp:
        def write_geom(geom_, fid_):
            fp.write({
                'geometry': mapping(geom_),
                'properties': {'fid': fid_},
            })

        if geom.type == 'Polygon':
            write_geom(geom, 0)
        else:
            for fid, poly in enumerate(list(geom)):
                write_geom(poly, fid)


def polygonize(mask, epsilon=1., min_area=10.):
    # https://www.programcreek.com/python/example/70440/cv2.findContours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    approx_contours = contours
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def create_model(input_shape):
    kernel_initializer = 'he_normal'

    inputs = Input(input_shape)

    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu',
               strides=(1, 1),
               padding='same',
               kernel_initializer=kernel_initializer
               )(inputs)

    x = BatchNormalization(momentum=0.01)(x)
    c1 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(c1)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # x, c2 = sat_conv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    c2 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(c2)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # x, c3 = sat_conv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    c3 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(c3)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # x, c4 = sat_conv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    c4 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(c4)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # x, c5 = sat_conv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    c5 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(c5)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # x = sat_upconv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2DTranspose(64, activation='relu', kernel_size=(3, 3), strides=(2, 2), padding='same',
                        output_padding=(1, 1))(x)

    x = concatenate([x, c5])
    # x = sat_upconv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2DTranspose(64, activation='relu', kernel_size=(3, 3), strides=(2, 2), padding='same',
                        output_padding=(1, 1))(x)

    x = concatenate([x, c4])
    # x = sat_upconv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2DTranspose(64, activation='relu', kernel_size=(3, 3), strides=(2, 2), padding='same',
                        output_padding=(1, 1))(x)

    x = concatenate([x, c3])
    # x = sat_upconv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2DTranspose(64, activation='relu', kernel_size=(3, 3), strides=(2, 2), padding='same',
                        output_padding=(1, 1))(x)

    x = concatenate([x, c2])
    # x = sat_upconv_block(x, conv_num=3)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2DTranspose(64, activation='relu', kernel_size=(3, 3), strides=(2, 2), padding='same',
                        output_padding=(1, 1))(x)

    x = concatenate([x, c1])
    # x, _ = sat_conv_block(x, conv_num=2)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
               kernel_initializer=kernel_initializer)(x)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    return Model(inputs=[inputs], outputs=[outputs])
