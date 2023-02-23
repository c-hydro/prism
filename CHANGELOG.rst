=========
Changelog
=========
Version 2.5.5 [2023-02-23]
**************************
UPD: **modified_conditional_merging.py**
    - Added support for WebDrops data download

UPD: **libs_model_griso_io.py**
    - Added support for WebDrops data download

Version 2.5.4 [2022-11-03]
**************************
FIX: **modified_conditional_merging.py**
    - Full support to backup GRISO only if gridded data not available

Version 2.5.3 [2022-07-26]
**************************
FIX: **modified_conditional_merging.py**
    - Added griso backup if gridded data not available
    - Fixed bug for sub-hourly drops2 aggregation

Version 2.5.2 [2022-07-04]
**************************
UPD: **modified_conditional_merging.py**
    - Added support to sub-hourly merging
    - Moved to prism repository

Version 2.5.1 [2022-03-02]
**************************
FIX: **modified_conditional_merging.py**
    - Fixed management of absence of point measurements

Version 2.5.0 [2021-10-26]
**************************
FIX: **modified_conditional_merging.py**
    - Added theoretical kernel estimation
    - Bug fixes, major system optimizations

FIX: **libs_model_griso_io.py**
    - Bug fixes, major system optimizations

FIX: **libs_model_griso_exec.py**
    - Added theoretical kernel estimation
    - Bug fixes, major system optimizations

Version 2.2.0 [2021-08-02]
**************************
UPD: **modified_conditional_merging.py**
    - Added support for AAIGrid inputs/outputs

UPD: **libs_model_griso_io.py**
    - Added support for AAIGrid inputs/outputs

Version 2.1.0 [2021-06-02]
**************************
UPD: **modified_conditional_merging.py**
    - Added beta support for radar rainfall products
    - Implemented tif inputs, implemented not-standard point files input.
    - Add netrc support for drops2. Bug fixes.

UPD: **libs_model_griso_io.py**
    - Add tif input/output routine
    - Add drops2 authentication support with netrc

Version 2.0.0 [2021-04-25]
**************************
UPD: **modified_conditional_merging.py**
    - Dynamic radius for GRISO implemented
    - Added support to point rain files (for FloodProofs compatibility)
    - Script structure fully revised and bug fixes

UPD: **libs_model_griso_exec.py**
    - Add dynamic radius support

Version 1.5.0 [2021-03-12]
**************************
FIX: **modified_conditional_merging.py**
    - Geotiff output implementation
    - Script structure updates
    - Various bug fixes and improvements

LIB: **libs_model_griso_exec.py**
    - Implemented for collecting the griso core functions

LIB: **libs_model_griso_generic.py**
    - Implemented for collecting the griso basic and geo functions

LIB: **libs_model_griso_io.py**
    - Implemented for collecting the griso input/output functions

Version 1.2.0 [2020-12-09]
**************************
UPD: **modified_conditional_merging.py**
    - Integrated with local station data configuration
    - Settings file implemented

Version 1.1.0 [2020-07-16]
**************************
UPD: **modified_conditional_merging.py**
    - Integrated with drops2 libraries. Updates and bug fixes

Version 1.0.0 [2020-03-26]
**************************
APP: **modified_conditional_merging.py**
    - Beta release for FloodProofs Bolivia

