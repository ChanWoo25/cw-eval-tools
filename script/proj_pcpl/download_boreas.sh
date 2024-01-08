#!/bin/bash
root=/data/datasets/dataset_boreas


#************************#
#* Section 1 (Total 16) *#
#************************#
aws s3 sync s3://boreas/boreas-2020-11-26-13-58  $root/boreas-2020-11-26-13-58 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request

# aws s3 sync s3://boreas/boreas-2020-12-04-14-00  $root/boreas-2020-12-04-14-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-26-10-59  $root/boreas-2021-01-26-10-59 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-26-11-22  $root/boreas-2021-01-26-11-22 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-02-09-12-55  $root/boreas-2021-02-09-12-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-09-14-23  $root/boreas-2021-03-09-14-23 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request

aws s3 sync s3://boreas/boreas-2021-04-22-15-00  $root/boreas-2021-04-22-15-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request

# aws s3 sync s3://boreas/boreas-2021-06-29-18-53  $root/boreas-2021-06-29-18-53 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-06-29-20-43  $root/boreas-2021-06-29-20-43 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-08-21-00  $root/boreas-2021-09-08-21-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request

aws s3 sync s3://boreas/boreas-2021-09-09-15-28  $root/boreas-2021-09-09-15-28 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2021-09-14-20-00  $root/boreas-2021-09-14-20-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2021-10-05-15-35  $root/boreas-2021-10-05-15-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2021-10-26-12-35  $root/boreas-2021-10-26-12-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2021-11-06-18-55  $root/boreas-2021-11-06-18-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2021-11-28-09-18  $root/boreas-2021-11-28-09-18 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request


#************************#
#* Section 2 (Total 28) *#
#************************#
# aws s3 sync s3://boreas/boreas-2020-12-01-13-26  $root/boreas-2020-12-01-13-26 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2020-12-18-13-44  $root/boreas-2020-12-18-13-44 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-15-12-17  $root/boreas-2021-01-15-12-17 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-19-15-08  $root/boreas-2021-01-19-15-08 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-02-02-14-07  $root/boreas-2021-02-02-14-07 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-02-13-38  $root/boreas-2021-03-02-13-38 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-23-12-43  $root/boreas-2021-03-23-12-43 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-30-14-23  $root/boreas-2021-03-30-14-23 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-08-12-44  $root/boreas-2021-04-08-12-44 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-13-14-49  $root/boreas-2021-04-13-14-49 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-15-18-55  $root/boreas-2021-04-15-18-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-20-14-11  $root/boreas-2021-04-20-14-11 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-29-15-55  $root/boreas-2021-04-29-15-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-05-06-13-19  $root/boreas-2021-05-06-13-19 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-05-13-16-11  $root/boreas-2021-05-13-16-11 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-06-03-16-00  $root/boreas-2021-06-03-16-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-06-17-17-52  $root/boreas-2021-06-17-17-52 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-07-20-17-33  $root/boreas-2021-07-20-17-33 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-07-27-14-43  $root/boreas-2021-07-27-14-43 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-08-05-13-34  $root/boreas-2021-08-05-13-34 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-02-11-42  $root/boreas-2021-09-02-11-42 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-07-09-35  $root/boreas-2021-09-07-09-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-10-15-12-35  $root/boreas-2021-10-15-12-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-10-22-11-36  $root/boreas-2021-10-22-11-36 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-02-11-16  $root/boreas-2021-11-02-11-16 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-14-09-47  $root/boreas-2021-11-14-09-47 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-16-14-10  $root/boreas-2021-11-16-14-10 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-23-14-27  $root/boreas-2021-11-23-14-27 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request



#************************#
#* All lists (Total 44) *#
#************************#
# aws s3 sync s3://boreas/boreas-2020-11-26-13-58  $root/boreas-2020-11-26-13-58 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2020-12-01-13-26  $root/boreas-2020-12-01-13-26 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2020-12-04-14-00  $root/boreas-2020-12-04-14-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2020-12-18-13-44  $root/boreas-2020-12-18-13-44 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-15-12-17  $root/boreas-2021-01-15-12-17 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-19-15-08  $root/boreas-2021-01-19-15-08 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-26-10-59  $root/boreas-2021-01-26-10-59 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-01-26-11-22  $root/boreas-2021-01-26-11-22 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-02-02-14-07  $root/boreas-2021-02-02-14-07 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-02-09-12-55  $root/boreas-2021-02-09-12-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-02-13-38  $root/boreas-2021-03-02-13-38 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-09-14-23  $root/boreas-2021-03-09-14-23 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-23-12-43  $root/boreas-2021-03-23-12-43 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-03-30-14-23  $root/boreas-2021-03-30-14-23 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-08-12-44  $root/boreas-2021-04-08-12-44 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-13-14-49  $root/boreas-2021-04-13-14-49 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-15-18-55  $root/boreas-2021-04-15-18-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-20-14-11  $root/boreas-2021-04-20-14-11 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-22-15-00  $root/boreas-2021-04-22-15-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-04-29-15-55  $root/boreas-2021-04-29-15-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-05-06-13-19  $root/boreas-2021-05-06-13-19 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-05-13-16-11  $root/boreas-2021-05-13-16-11 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-06-03-16-00  $root/boreas-2021-06-03-16-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-06-17-17-52  $root/boreas-2021-06-17-17-52 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-06-29-18-53  $root/boreas-2021-06-29-18-53 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-06-29-20-43  $root/boreas-2021-06-29-20-43 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-07-20-17-33  $root/boreas-2021-07-20-17-33 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-07-27-14-43  $root/boreas-2021-07-27-14-43 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-08-05-13-34  $root/boreas-2021-08-05-13-34 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-02-11-42  $root/boreas-2021-09-02-11-42 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-07-09-35  $root/boreas-2021-09-07-09-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-08-21-00  $root/boreas-2021-09-08-21-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-09-15-28  $root/boreas-2021-09-09-15-28 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-09-14-20-00  $root/boreas-2021-09-14-20-00 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-10-05-15-35  $root/boreas-2021-10-05-15-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-10-15-12-35  $root/boreas-2021-10-15-12-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-10-22-11-36  $root/boreas-2021-10-22-11-36 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-10-26-12-35  $root/boreas-2021-10-26-12-35 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-02-11-16  $root/boreas-2021-11-02-11-16 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-06-18-55  $root/boreas-2021-11-06-18-55 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-14-09-47  $root/boreas-2021-11-14-09-47 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-16-14-10  $root/boreas-2021-11-16-14-10 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-23-14-27  $root/boreas-2021-11-23-14-27 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
# aws s3 sync s3://boreas/boreas-2021-11-28-09-18  $root/boreas-2021-11-28-09-18 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
