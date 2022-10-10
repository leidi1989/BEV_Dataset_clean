#!/usr/bin/python3
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
import utm
import re
from pyproj import Proj

origin_lat = 0.0
origin_lon = 0.0
origin_x = 0.0
origin_y = 0.0
zone_number = 0
zone_letter = ''

mercator = None


def SetOrigin_Utm(lat: float, lon: float):
    """to set the origin_lat origin_lon origin_x origin_y and zone number"""
    global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    x, y, num, letter = utm.from_latlon(lat, lon)
    origin_lat = lat
    origin_lon = lon
    origin_x = x
    origin_y = y
    zone_number = num
    zone_letter = letter


def ll2xy_Utm(lat: float, lon: float):
    """latitude and longtitude to x and y"""
    global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    easting, northing, _, _ = utm.from_latlon(
        lat, lon, zone_number, zone_letter)
    x = easting - origin_x
    y = northing - origin_y
    return x, y


def xy2ll_Utm(x: float, y: float):
    """x and y to latitude and longtitude"""
    global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    easting = x+origin_x
    northing = y + origin_y
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    return lat, lon

def SetOrigin_Mercator(lat: float, lon: float):
    global mercator
    lat_string=str(lat)
    lon_string=str(lon)
    proj_string='+proj=tmerc +lat_0='+lat_string+ ' +lon_0='+lon_string+' +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    mercator=Proj(proj_string, preserve_units=False)

def ll2xy_Mercator(lat:float,lon:float):
    global mercator
    x,y=mercator(lon,lat)
    return x,y

def xy2ll_Mercator(x:float,y:float):
    global mercator
    lon,lat=mercator(x,y,inverse=True)
    return lat,lon
