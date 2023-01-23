\timing

-- LOAD FILE TO BE MERGED
DROP TABLE merge.panoids_2011_staging;
CREATE TABLE merge.panoids_2011_staging(
    index integer,
    xcoord double precision, 
    ycoord double precision, 
    panoid varchar(30),
    lat double precision, 
    lon double precision,
    month integer, 
    year integer,
    azi double precision,
    id varchar(40), 
    idx integer
);

copy merge.panoids_2011_staging
FROM PROGRAM 'tail -n +2 /home/emily/phd/0_get_images/outputs/psql/census_2011/census_2011_image_meta_single.csv' DELIMITER as ',';

-- CREATE GEOGRAPHY
DROP TABLE merge.panoids_2011;
CREATE TABLE merge.panoids_2011(
    panoid varchar(30),
    lat double precision, 
    lon double precision,
    geom geometry(Point, 4326),
    month integer, 
    year integer,
    azi double precision,
    id varchar(40),
    idx integer
);

INSERT INTO merge.panoids_2011 (panoid, lat, lon, geom, month, year, azi, id, idx)
SELECT
    panoid,
    lat,
    lon,
    -- geometry::Point(lat, lon, 4326) as geom,
    ST_SetSRID(ST_MakePoint(lon,lat), 4326) as geom,
    month,
    year,
    azi,
    id, 
    idx
FROM merge.panoids_2011_staging;

-- DELETE INDEX idx_panoids_2011;
CREATE INDEX idx_panoids_2011 on merge.panoids_2011 USING gist(geom);

-- -- LOAD OUTPUT FILE
-- -- shp2pgsql -s 4326 OA_2011_London_gen_MHW.shp london postgres > oa.sql
-- -- psql -U emily -d london -f oa.sql

-- DELETE INDEX idx_oa;
CREATE INDEX idx_oa on poly USING gist(geom);


\COPY (SELECT points.panoid AS panoid, points.lat AS lat, points.lon as lon, points.idx as idx, oa.oa11cd AS oa_name FROM merge.panoids_2011 AS points  JOIN public.london AS oa ON ST_Contains(oa.geom, points.geom)) TO '/media/emily/south/chapter3data/outputs/2011_panoids_merged_to_oa.csv' csv header;


-- \COPY (
-- SELECT 
--     points.panoid AS panoid,
--     oa.oa11cd AS oa_name
-- FROM merge.panoids_2011 AS points 
-- JOIN public.london AS oa 
-- ON ST_Contains(oa.geom, points.geom)
-- WHERE oa.oa11cd = 'E00000001'
-- )
-- TO '/media/emily/south/chapter3data/outputs/2011_panoids_merged_to_oa.csv' csv header;

-- SELECT *
-- FROM merge.panoids_2011, poly
-- WHERE ST_Contains(poly.geom, merge.panoids_2011.geom)

-- CREATE TABLE poly AS                   
-- /                

-- # convert SRID 
-- https://www.cadlinecommunity.co.uk/hc/en-us/articles/209661629-PostGIS-How-can-you-find-the-CRS-SRID-of-your-Spatial-Data-
