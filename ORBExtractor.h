#ifndef ORB_EXTRACTOR_H
#define ORB_EXTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <list>

namespace Goudan
{
    class ExtractorNode
    {
    public:
        ExtractorNode() : bNoMore(false){};
        // 分裂节点为四个，四叉树
        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        // 这个节点区域内的提取出来的ORB关键点
        std::vector<cv::KeyPoint> vKeys;
        // 描述此节点的区域
        cv::Point2i UL, UR, BL, BR;
        // 一个迭代器 方便遍历?
        std::list<ExtractorNode>::iterator lit;
        // 是否需要更多？(当只有一个的时候为true)
        bool bNoMore;
    };

    class ORBExtractor
    {
    public:
        ORBExtractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
        ~ORBExtractor() {}

        // 重载了()运算符，作为提取器的对外接口  提取比较分散的ORB
        void operator()(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints, cv::OutputArray desciptors);

        int inline GetLevels() { return nlevels; }
        float inline GetScaleFactor() { return scaleFactor; }
        std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }
        std::vector<float> inline GetInverseScaleFactors() { return mvInvScaleFactor; }
        std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }
        std::vector<float> inline GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }
        // 保存图像金字塔
        std::vector<cv::Mat> mvImagePyramid;

    protected:
        // 计算高斯金字塔
        void ComputePyramid( cv::Mat image );
        // 计算关键点并用四叉树进行存储
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
        // 为关键点分配四叉树
        std::vector< cv::KeyPoint > DistributeOctTree( const std::vector< cv::KeyPoint >& vToDistributeKeys, 
      const int& minX, const int& maxX, const int& minY, const int& maxY, const int& nFeatures, const int& level );
        //
        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);


        std::vector<cv::Point> pattern; // 保存BRIFE描述子的模式

        int nfeatures;      // 总共需要的特征点数
        double scaleFactor; // 每层之间的缩放因子
        int nlevels;        // 金字塔层数
        int iniThFAST;      // 初始化的FAST角点的阈值
        int minThFAST;      // 最小的FAST角点的阈值

        std::vector<int> mnFeaturesPerLevel; // 计算出来的每层所需的特征点数
        // to set the local block border
        std::vector<int> umax;               // Patch圆的最大坐标
        std::vector<float> mvScaleFactor;    // 累乘得到的每层相对第一层的尺度因子
        std::vector<float> mvInvScaleFactor; // 缩放因子的逆
        std::vector<float> mvLevelSigma2;    // 尺度因子mvScaleFactor的平方
        std::vector<float> mvInvLevelSigma2; // 尺度因子mvScaleFactor的平方的逆
    };

}

#endif