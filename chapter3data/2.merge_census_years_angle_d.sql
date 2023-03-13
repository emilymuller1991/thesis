\timing

-- -- LOAD FILE TO BE MERGED
DROP TABLE merge.panoids_2021_staging;
CREATE TABLE merge.panoids_2021_staging(
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
    idx integer,
    idxy varchar(10),
    keep integer
);

copy merge.panoids_2021_staging
FROM PROGRAM 'tail -n +2 /home/emily/phd/0_get_images/outputs/psql/census_2021/census_2021_image_meta_double_dropna_d_matched_centroid.csv' DELIMITER as ',';

-- CREATE GEOGRAPHY
DROP TABLE merge.panoids_2021;
CREATE TABLE merge.panoids_2021(
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

INSERT INTO merge.panoids_2021 (panoid, lat, lon, geom, month, year, azi, id, idx)
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
FROM merge.panoids_2021_staging;

-- DELETE INDEX idx_panoids_2011;
CREATE INDEX idx_panoids_2021 on merge.panoids_2021 USING gist(geom);
CREATE INDEX idx_merged_2011 on joins.census_2011_buffered_dropna_d_matched_centroid USING gist(geom);

-- -- LOAD OUTPUT FILE
-- -- shp2pgsql -s 4326 OA_2011_London_gen_MHW.shp london postgres > oa.sql
-- -- psql -U emily -d london -f oa.sql

-- DELETE INDEX idx_oa;
-- CREATE INDEX idx_oa on poly USING gist(geom);


\COPY (SELECT points.panoid AS panoid_2021, points.lat AS lat_2021, points.lon as lon_2021, points.idx as idx_2021, buffered.panoid AS panoid_2011, buffered.lat AS lat_2011, buffered.lon as lon_2011, buffered.idx as idx_2011 FROM merge.panoids_2021 AS points  JOIN joins.census_2011_buffered_dropna_d_matched_centroid AS buffered ON ST_Contains(buffered.geom, points.geom)) TO 'outputs/2011_buffered_merge_to_2021_dropna_d_matched_centroid.csv' csv header;
