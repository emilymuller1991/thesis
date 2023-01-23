\timing

-- LOAD FILE TO BE MERGED
DROP TABLE merge.panoids_2018_staging;
CREATE TABLE merge.panoids_2018_staging(
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

copy merge.panoids_2018_staging
FROM PROGRAM 'tail -n +2 /home/emily/phd/0_get_images/outputs/psql/2018/greater_london_2018_panoids_to_download.csv' DELIMITER as ',';

-- CREATE GEOGRAPHY
DROP TABLE merge.panoids_2018;
CREATE TABLE merge.panoids_2018(
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

INSERT INTO merge.panoids_2018 (panoid, lat, lon, geom, month, year, azi, id, idx)
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
FROM merge.panoids_2018_staging;

-- DELETE INDEX idx_panoids_2011;
CREATE INDEX idx_panoids_2018 on merge.panoids_2018 USING gist(geom);

-- -- LOAD OUTPUT FILE
-- -- shp2pgsql -s 4326 OA_2011_London_gen_MHW.shp london postgres > oa.sql
-- -- psql -U emily -d london -f oa.sql

-- DELETE INDEX idx_oa;
CREATE INDEX idx_oa on poly USING gist(geom);


\COPY (SELECT points.panoid AS panoid, points.lat AS lat, points.lon as lon, points.idx as idx, oa.oa11cd AS oa_name FROM merge.panoids_2018 AS points  JOIN public.london AS oa ON ST_Contains(oa.geom, points.geom)) TO '/media/emily/south/chapter3data/outputs/2018_panoids_merged_to_oa.csv' csv header;
