#include <iostream>
#include <opencv2/opencv.hpp>

#include <thread>
#include <future>

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstring>
#include <netinet/in.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <chrono>  // For timing FPS
#include <sstream> // For string formatting
#include <deque>   // For sliding window
#include "yolo11_seg.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#define SERVER_IP "192.168.137.1"
#define SERVER_PORT 8002
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 640
class FPSCounter {
public:
    FPSCounter(int window_seconds = 5) : window_seconds(window_seconds) {}

    void addFrame() {
        double now = cv::getTickCount() / cv::getTickFrequency();
        timestamps.push_back(now);

        // 移除窗口外的时间戳
        while (!timestamps.empty() && now - timestamps.front() > window_seconds) {
            timestamps.pop_front();
        }
    }

    double getFPS() {
        if (timestamps.size() < 2) return 0.0;
        return timestamps.size() / (timestamps.back() - timestamps.front());
    }

    void updateFPS() {
        double now = cv::getTickCount() / cv::getTickFrequency();
        if (now - last_update_time >= window_seconds) {
            last_update_time = now;
        }
    }

private:
    int window_seconds;
    std::deque<double> timestamps;
    double last_update_time = 0.0;
};
// 数据处理线程安全队列模板类
template<typename T>
class ThreadSafeQueue {
public:
    void push(const T &value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        cv_.notify_one();
    }

    bool pop(T &value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    void wait_and_pop2(T &value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return queue_.size() >= 2; });
        value = std::move(queue_.front());
        queue_.pop();
    }
    void wait_and_pop(T &value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty(); });
        value = std::move(queue_.front());
        queue_.pop();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

cv::Mat interpolate_frame( cv::Mat& img,
                          object_detect_result_list od_results1,
                          object_detect_result_list od_results) {

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = img.cols;
    src_image.height = img.rows;
    src_image.width_stride = img.step[0];  // 每行的字节数
    src_image.height_stride = img.step[1]; // 步幅，通常为1
    src_image.size = img.total() * img.elemSize();
    int width = src_image.width;
    int height = src_image.height;
    src_image.virt_addr = img.data;

    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.fd = -1;
    char *interpolated_frame = (char *)src_image.virt_addr;


    char *ori_img = (char *)src_image.virt_addr;
    unsigned char class_colors[][3] = {
            {255, 56, 56}, {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49}, {72, 249, 10},
            {146, 204, 23}, {61, 219, 134}, {26, 147, 52}, {0, 212, 187}, {44, 153, 168}, {0, 194, 255},
            {52, 69, 147}, {100, 115, 255}, {0, 24, 236}, {132, 56, 255}, {82, 0, 133}, {203, 56, 255},
            {255, 149, 200}, {255, 55, 199}
    };
    // if (od_results.count >= 1) {
    //     if (od_results.results_seg == nullptr || od_results.results_seg[0].seg_mask == nullptr) {
    //         printf("Segmentation mask is null!\n");
    //         return img;
    //     }
    //     seg_mask = od_results.results_seg[0].seg_mask;

    //     if (src_image.virt_addr == nullptr) {
    //         printf("Source image virtual address is null!\n");
    //         return img;
    //     }

    // printf("test3!\n");
    // // draw boxes
    // for (int i = 0; i < od_results.count; i++) {
    //     object_detect_result *od_result = &(od_results.results[i]);
    //     int x1 = od_result->box.left;
    //     int y1 = od_result->box.top;
    //     int x2 = od_result->box.right;
    //     int y2 = od_result->box.bottom;

    //     draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 3);
    //     char text[256];
    //     printf("%s", coco_cls_to_name(od_result->cls_id));
    //     sprintf(text, "%s %.1f%%", coco_cls_to_name(od_result->cls_id), od_result->prop * 100);
    //     draw_text(&src_image, text, x1, y1 - 16, COLOR_BLUE, 10);
    // }




    // 创建新的掩码，存储两帧的交集
    uint8_t *seg_mask;
    uint8_t *seg_mask1=od_results.results_seg[0].seg_mask;
    uint8_t *seg_mask2=od_results1.results_seg[0].seg_mask;
    // std::vector<uint8_t> combined_mask(width * height, 0);
    // for (int i = 0; i < width * height; i++) {
    //     seg_mask[i] = seg_mask1[i];
    // }

    // char *ori_img = (char *)src_image.virt_addr;
    int cls_id = od_results.results[0].cls_id;

    float alpha = 0.5f; // opacity

    for (int j = 0; j < height; j++) {
        for (int k = 0; k < width; k++) {
            int pixel_offset = 3 * (j * width + k);
            if (seg_mask1[j * width + k] != 0 & seg_mask2[j * width + k] != 0) {
                    ori_img[pixel_offset + 0] = (unsigned char)clamp(class_colors[seg_mask1[j * width + k] % N_CLASS_COLORS][0] * (1 - alpha) + ori_img[pixel_offset + 0] * alpha, 0, 255); // r
                    ori_img[pixel_offset + 1] = (unsigned char)clamp(class_colors[seg_mask1[j * width + k] % N_CLASS_COLORS][1] * (1 - alpha) + ori_img[pixel_offset + 1] * alpha, 0, 255); // g
                    ori_img[pixel_offset + 2] = (unsigned char)clamp(class_colors[seg_mask1[j * width + k] % N_CLASS_COLORS][2] * (1 - alpha) + ori_img[pixel_offset + 2] * alpha, 0, 255); // b
            }
        }
    }
    



    // // 在原图上绘制交集掩码
    // float alpha = 0.5f;
    // for (int j = 0; j < height; j++) {
    //     for (int k = 0; k < width; k++) {
    //         int pixel_offset = 3 * (j * width + k);
    //         if (combined_mask[j * width + k] != 0) {
    //             interpolated_frame[pixel_offset + 0] =
    //                 (unsigned char)clamp(class_colors[combined_mask[j * width + k] % N_CLASS_COLORS][0] * (1 - alpha) +
    //                                      interpolated_frame[pixel_offset + 0] * alpha, 0, 255);
    //             interpolated_frame[pixel_offset + 1] =
    //                 (unsigned char)clamp(class_colors[combined_mask[j * width + k] % N_CLASS_COLORS][1] * (1 - alpha) +
    //                                      interpolated_frame[pixel_offset + 1] * alpha, 0, 255);
    //             interpolated_frame[pixel_offset + 2] =
    //                 (unsigned char)clamp(class_colors[combined_mask[j * width + k] % N_CLASS_COLORS][2] * (1 - alpha) +
    //                                      interpolated_frame[pixel_offset + 2] * alpha, 0, 255);
    //         }
    //     }
    // }

    // 平均化识别框
    for (size_t i = 0; i < std::min(od_results1.count, od_results.count); i++) {
        object_detect_result *od1 = &(od_results1.results[i]);
        object_detect_result *od2 = &(od_results.results[i]);

        // // 计算坐标的改变量
        int delta_x1 = std::abs(od1->box.left - od2->box.left);
        int delta_y1 = std::abs(od1->box.top - od2->box.top);
        int delta_x2 = std::abs(od1->box.right - od2->box.right);
        int delta_y2 = std::abs(od1->box.bottom - od2->box.bottom);

        // // 判断坐标改变量是否小于40
        int avg_x1, avg_y1, avg_x2, avg_y2;
        if (delta_x1 < 30 && delta_y1 < 30 && delta_x2 < 30 && delta_y2 < 30) {
            // 如果每个坐标的改变量都小于40，取平均值
            avg_x1 = (od1->box.left + od2->box.left) / 2;
            avg_y1 = (od1->box.top + od2->box.top) / 2;
            avg_x2 = (od1->box.right + od2->box.right) / 2;
            avg_y2 = (od1->box.bottom + od2->box.bottom) / 2;
        } else {
            // 否则，取第二个框的坐标
        avg_x1 = od2->box.left;
        avg_y1 = od2->box.top;
        avg_x2 = od2->box.right;
        avg_y2 = od2->box.bottom;
        }

        // 绘制矩形
        draw_rectangle(&src_image, avg_x1, avg_y1, avg_x2 - avg_x1, avg_y2 - avg_y1, COLOR_RED, 3);

        // 绘制文本
        char text[256];
        sprintf(text, "%s %.1f%%", coco_cls_to_name(od2->cls_id), (od2->prop + od2->prop) * 50);
        draw_text(&src_image, text, avg_x1, avg_y1 - 16, COLOR_BLUE, 10);
    }




    // for (int i = 0; i < od_results.count; i++) {
    //     object_detect_result *od_result = &(od_results.results[i]);
    //     int x1 = od_result->box.left;
    //     int y1 = od_result->box.top;
    //     int x2 = od_result->box.right;
    //     int y2 = od_result->box.bottom;

    //     draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 3);
    //     char text[256];
    //     printf("%s", coco_cls_to_name(od_result->cls_id));
    //     sprintf(text, "%s %.1f%%", coco_cls_to_name(od_result->cls_id), od_result->prop * 100);
    //     draw_text(&src_image, text, x1, y1 - 16, COLOR_BLUE, 10);
    // }
    return img;
}// 推理函数声明
cv::Mat rknn_segment_inferd(rknn_app_context_t& rknn_app_ctx, cv::Mat& img,cv::Mat& img2, object_detect_result_list *last_od_results) 
{

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    printf("test1!\n");


    src_image.width = img.cols;
    src_image.height = img.rows;
    src_image.width_stride = img.step[0];  // 每行的字节数
    src_image.height_stride = img.step[1]; // 步幅，通常为1
    src_image.size = img.total() * img.elemSize();
    src_image.virt_addr = img.data;

    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.fd = -1;
    uint8_t *seg_mask;
    object_detect_result_list od_results;
    int ret = inference_yolo11_seg_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0) {
        printf("Inference failed! ret=%d\n", ret);
        return img;
    }
    unsigned char class_colors[][3] = {
            {255, 56, 56}, {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49}, {72, 249, 10},
            {146, 204, 23}, {61, 219, 134}, {26, 147, 52}, {0, 212, 187}, {44, 153, 168}, {0, 194, 255},
            {52, 69, 147}, {100, 115, 255}, {0, 24, 236}, {132, 56, 255}, {82, 0, 133}, {203, 56, 255},
            {255, 149, 200}, {255, 55, 199}};
    // draw mask  
    printf("test2!\n");
    if (od_results.count >= 1) {
        if (od_results.results_seg == nullptr || od_results.results_seg[0].seg_mask == nullptr) {
            printf("Segmentation mask is null!\n");
            return img;
        }
        seg_mask = od_results.results_seg[0].seg_mask;

        if (src_image.virt_addr == nullptr) {
            printf("Source image virtual address is null!\n");
            return img;
        }
        int width = src_image.width;
        int height = src_image.height;
        char *ori_img = (char *)src_image.virt_addr;
        int cls_id = od_results.results[0].cls_id;

        float alpha = 0.5f; // opacity

        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                int pixel_offset = 3 * (j * width + k);
                if (seg_mask[j * width + k] != 0) {
                    ori_img[pixel_offset + 0] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][0] * (1 - alpha) + ori_img[pixel_offset + 0] * alpha, 0, 255); // r
                    ori_img[pixel_offset + 1] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][1] * (1 - alpha) + ori_img[pixel_offset + 1] * alpha, 0, 255); // g
                    ori_img[pixel_offset + 2] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][2] * (1 - alpha) + ori_img[pixel_offset + 2] * alpha, 0, 255); // b
                }
            }
        }


    }
    printf("test3!\n");
    // draw boxes
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result *od_result = &(od_results.results[i]);
        int x1 = od_result->box.left;
        int y1 = od_result->box.top;
        int x2 = od_result->box.right;
        int y2 = od_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 3);
        char text[256];
        printf("%s", coco_cls_to_name(od_result->cls_id));
        sprintf(text, "%s %.1f%%", coco_cls_to_name(od_result->cls_id), od_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 16, COLOR_BLUE, 10);
    }
    printf("test4!\n");
    cv::Mat ret1 = interpolate_frame(img2, *last_od_results, od_results);

    printf("test5!\n");
    *last_od_results = od_results;
    // cv::cvtColor(ret1, ret1, cv::COLOR_RGB2BGR);
    // output_queue.push(ret1);
    // free(last_mask);
    // free(last_od_results);
    // 返回 RGB 格式的图像
    return ret1;
}
// 推理函数声明
cv::Mat rknn_segment_infer(rknn_app_context_t& rknn_app_ctx, cv::Mat& img,object_detect_result_list *last_od_results) {
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    src_image.width = img.cols;
    src_image.height = img.rows;
    src_image.width_stride = img.step[0];  // 每行的字节数
    src_image.height_stride = img.step[1]; // 步幅，通常为1
    src_image.size = img.total() * img.elemSize();
    
    src_image.virt_addr = img.data;

    src_image.format = IMAGE_FORMAT_RGB888;

    src_image.fd = -1;
    uint8_t *seg_mask;
    object_detect_result_list od_results;
    int ret = inference_yolo11_seg_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0) {
        printf("Inference failed! ret=%d\n", ret);
        return img;
    }

    // draw mask
    if (od_results.count >= 1) {
        unsigned char class_colors[][3] = {
            {255, 56, 56}, {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49}, {72, 249, 10},
            {146, 204, 23}, {61, 219, 134}, {26, 147, 52}, {0, 212, 187}, {44, 153, 168}, {0, 194, 255},
            {52, 69, 147}, {100, 115, 255}, {0, 24, 236}, {132, 56, 255}, {82, 0, 133}, {203, 56, 255},
            {255, 149, 200}, {255, 55, 199}
        };
        if (od_results.results_seg == nullptr || od_results.results_seg[0].seg_mask == nullptr) {
            printf("Segmentation mask is null!\n");
            return img;
        }


        if (src_image.virt_addr == nullptr) {
            printf("Source image virtual address is null!\n");
            return img;
        }
        int width = src_image.width;
        int height = src_image.height;
        char *ori_img = (char *)src_image.virt_addr;
        int cls_id = od_results.results[0].cls_id;
        seg_mask = od_results.results_seg[0].seg_mask;
        float alpha = 0.5f; // opacity
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                int pixel_offset = 3 * (j * width + k);
                if (seg_mask[j * width + k] != 0) {
                    ori_img[pixel_offset + 0] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][0] * (1 - alpha) + ori_img[pixel_offset + 0] * alpha, 0, 255); // r
                    ori_img[pixel_offset + 1] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][1] * (1 - alpha) + ori_img[pixel_offset + 1] * alpha, 0, 255); // g
                    ori_img[pixel_offset + 2] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][2] * (1 - alpha) + ori_img[pixel_offset + 2] * alpha, 0, 255); // b
                }
            }
        }
        // free(seg_mask);
    }

    // draw boxes
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result *det_result = &(od_results.results[i]);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 3);
        char text[256];
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 16, COLOR_BLUE, 10);
    }

    *last_od_results = od_results;

    // 返回 RGB 格式的图像
    return img;

}



// 图像处理线程
void process_frames(rknn_app_context_t &rknn_app_ctx, ThreadSafeQueue<cv::Mat> &input_queue,
                    ThreadSafeQueue<cv::Mat> &output_queue) {
    cv::Mat frame;
    cv::Mat frame2;

    int i=0;
    object_detect_result_list last_od_results;
    while (true) {
        printf("Inference!\n");
        if (i==0){
        input_queue.wait_and_pop(frame);
        cv::Mat processed_frame = rknn_segment_infer(rknn_app_ctx, frame,&last_od_results);
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        output_queue.push(frame);
        // i++;
        }
        else{       
         
        input_queue.wait_and_pop(frame);
   
        input_queue.wait_and_pop(frame2);
        if (frame.empty()) break;  // 结束信号 
      
        cv::Mat processed_frame = rknn_segment_inferd(rknn_app_ctx, frame,frame2,&last_od_results);
        cv::cvtColor(frame,frame, cv::COLOR_RGB2BGR);

        cv::cvtColor(frame2,frame2, cv::COLOR_RGB2BGR);
        output_queue.push(frame);
        output_queue.push(frame2);}

    }
}

// 图像发送线程
void send_frames(ThreadSafeQueue<cv::Mat> &output_queue, int sock) {
    cv::Mat frame;
    FPSCounter fpsCounter;
    while (true) {
        output_queue.wait_and_pop(frame);
        if (frame.empty()) break;  // 结束信号
        // fpsCounter.addFrame();
        // fpsCounter.updateFPS();

        // 显示 FPS
        // double fps = fpsCounter.getFPS();
        // std::ostringstream oss;
        // oss << "FPS: " << std::fixed << std::setprecision(2) << fps;
        // std::string fps_text = oss.str();
        // cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame, buf);
        send(sock, buf.data(), buf.size(), 0);
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }
}

int init_socket() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed!\n";
        return -1;
    }

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);

    if (connect(sock, (sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection failed!\n";
        return -1;
    }

    return sock;
}

int main(int argc, char **argv) {
    // if (argc != 2) {
    //     std::cerr << argv[0] << " <model_path>\n";
    //     return -1;
    // }

    const char *model_path = argv[1];
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    if (init_yolo11_seg_model(model_path, &rknn_app_ctx) != 0) {
        std::cerr << "Model initialization failed!\n";
        return -1;
    }

    int sock = init_socket();
    // if (sock < 0) return -1;

    cv::VideoCapture cap(0);
    // if (!cap.isOpened()) {
    //     std::cerr << "Failed to open camera!\n";
    //     return -1;
    // }
    int fps = 5; // 目标帧率
    // cap.set(cv::CAP_PROP_FPS, fps);

    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
    // cap(0).set(cv::CAP_PROP_FPS, 8);
    ThreadSafeQueue<cv::Mat> input_queue;
    ThreadSafeQueue<cv::Mat> output_queue;

    FPSCounter fpsCounter;

    std::thread processing_thread(process_frames, std::ref(rknn_app_ctx), std::ref(input_queue), std::ref(output_queue));
    std::thread sending_thread(send_frames, std::ref(output_queue), sock);
    init_post_process();
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        int frame_height = frame.rows;
        int frame_width = frame.cols;
        // std::cout << "Frame size: " << frame_width << "x" << frame_height << std::endl;

        cv::Mat resized_img;
        
        
        cv::resize(frame, resized_img, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
        // cv::rotate(resized_img, resized_img, cv::ROTATE_90_COUNTERCLOCKWISE);
        // cv::cvtColor(frame,frame, cv::COLOR_BGR2RGB);

        input_queue.push(resized_img);
        // input_queue.push(frame);
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (cv::waitKey(1) == 27) break;  // ESC 键退出
    }

    input_queue.push(cv::Mat());  // 结束信号
    output_queue.push(cv::Mat());

    processing_thread.join();
    sending_thread.join();
    deinit_post_process();
    close(sock);
    release_yolo11_seg_model(&rknn_app_ctx);

    return 0;
}
