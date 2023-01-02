#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

int main(int argc,char** argv)
{
        PointCloudT::Ptr cloud(new PointCloudT);
		
		// 读取点云
        std::string filename = "dynamic.ply";
        if (pcl::io::loadPLYFile(filename, *cloud)<0)
        {
            std::cout << "error";
        }

		// 点云显示
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (0, 0, 0);
        viewer->addPointCloud<PointT> (cloud, "sample cloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
        viewer->initCameraParameters ();
        while (!viewer->wasStopped())
        {
            viewer->spinOnce();
        }

        return 0;
}