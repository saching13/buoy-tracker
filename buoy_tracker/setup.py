from setuptools import find_packages, setup
import os

package_name = 'buoy_tracker'

package_data_files = []
shared_package_path = 'share/' + package_name

for (dirpath, dirnames, filenames) in os.walk('resource'):
    dest_path = shared_package_path + '/' + dirpath 
    files_list = []
    for file in filenames:
        files_list.append(os.path.join(dirpath, file))
    package_data_files.append((dest_path, files_list))

# print("Printing the package data files list --------------> ")
# print(package_data_files)
# print("---")
package_data_files.append((shared_package_path, ['package.xml']))
package_data_files.append((shared_package_path, ['launch/buoy_tracker.launch.py']))
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=package_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sachin',
    maintainer_email='sachin.guruswamy@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'buoy_tracker_node = buoy_tracker.buoy_tracker_node:main'
        ],
    },
)
