#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QThread>
#include <QMutex>
#include <QKeyEvent>
#include <QCursor>
#include <QScreen>
#include <QPainter>
#include <QPushButton>
#include <QMessageBox>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

// ---------------------------------------------------------
// Camera Worker Thread
// ---------------------------------------------------------
class CameraWorker : public QObject {
    Q_OBJECT
public:
    CameraWorker(QObject *parent = nullptr) : QObject(parent) {}

public slots:
    void startCamera() {
        cv::VideoCapture cap(0); // Open default camera
        if (!cap.isOpened()) {
            emit errorOccurred("Failed to open camera!");
            return;
        }
        
        // Set resolution for better tracking
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        cv::Mat frame;
        while (m_running) {
            cap >> frame;
            if (frame.empty()) break;

            emit frameReady(frame.clone());
            QThread::msleep(10); // ~100 FPS cap to save CPU
        }
        cap.release();
    }

    void stop() { m_running = false; }

signals:
    void frameReady(const cv::Mat &frame);
    void errorOccurred(const QString &msg);

private:
    std::atomic<bool> m_running{true};
};

// ---------------------------------------------------------
// Main Lightgun Widget
// ---------------------------------------------------------
class LightgunWidget : public QWidget {
    Q_OBJECT
public:
    LightgunWidget(QWidget *parent = nullptr) : QWidget(parent) {
        setWindowTitle("Qt/OpenCV Lightgun Emulator");
        resize(800, 600);

        // UI Setup
        QVBoxLayout *layout = new QVBoxLayout(this);
        m_videoLabel = new QLabel(this);
        m_videoLabel->setAlignment(Qt::AlignCenter);
        m_videoLabel->setStyleSheet("background-color: black;");
        layout->addWidget(m_videoLabel);

        m_statusLabel = new QLabel("Status: Click the 4 corners of your TV (Top-Left, Top-Right, Bottom-Right, Bottom-Left). Press 'R' to reset.", this);
        m_statusLabel->setStyleSheet("color: white; font-weight: bold; padding: 5px;");
        layout->addWidget(m_statusLabel);

        // Setup Camera Thread
        m_cameraThread = new QThread(this);
        m_cameraWorker = new CameraWorker();
        m_cameraWorker->moveToThread(m_cameraThread);

        connect(m_cameraThread, &QThread::started, m_cameraWorker, &CameraWorker::startCamera);
        connect(m_cameraThread, &QThread::finished, m_cameraWorker, &QObject::deleteLater);
        connect(m_cameraWorker, &CameraWorker::frameReady, this, &LightgunWidget::processFrame);
        connect(m_cameraWorker, &CameraWorker::errorOccurred, this, &LightgunWidget::handleError);

        m_cameraThread->start();

        // Screen resolution for mouse mapping
        QScreen *screen = QGuiApplication::primaryScreen();
        m_screenWidth = screen->size().width();
        m_screenHeight = screen->size().height();
    }

    ~LightgunWidget() {
        m_cameraWorker->stop();
        m_cameraThread->quit();
        m_cameraThread->wait();
    }

protected:
    void processFrame(const cv::Mat &frame) {
        cv::Mat displayFrame = frame.clone();
        cv::Mat gray, thresh;
        
        // 1. Detect the "Lightgun" (Bright IR/LED dot)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // High threshold to isolate only the brightest light source
        cv::threshold(gray, thresh, 230, 255, cv::THRESH_BINARY); 
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Point2f lightPos(-1, -1);
        double maxArea = 0;

        for (const auto &contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 10 && area < 5000 && area > maxArea) { // Filter noise
                maxArea = area;
                cv::Moments m = cv::moments(contour);
                if (m.m00 > 0) {
                    lightPos = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
                }
            }
        }

        // 2. Draw Calibration Points
        for (int i = 0; i < m_calibPoints.size(); ++i) {
            cv::circle(displayFrame, m_calibPoints[i], 10, cv::Scalar(0, 255, 0), -1);
            cv::putText(displayFrame, std::to_string(i + 1), m_calibPoints[i] + cv::Point(15, 15), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // 3. Map to Screen and Emulate Mouse
        if (m_isCalibrated && lightPos.x > 0) {
            // Apply Perspective Transform
            std::vector<cv::Point2f> src_pts = {lightPos};
            std::vector<cv::Point2f> dst_pts;
            cv::perspectiveTransform(src_pts, dst_pts, m_homography);

            int targetX = static_cast<int>(dst_pts[0].x);
            int targetY = static_cast<int>(dst_pts[0].y);

            // Clamp to screen boundaries
            targetX = std::max(0, std::min(m_screenWidth, targetX));
            targetY = std::max(0, std::min(m_screenHeight, targetY));

            // Move OS Mouse
            QCursor::setPos(targetX, targetY);

            // Draw crosshair on camera feed
            cv::circle(displayFrame, lightPos, 15, cv::Scalar(0, 0, 255), 2);
            cv::putText(displayFrame, "TRACKING", cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        } else if (lightPos.x > 0) {
            cv::circle(displayFrame, lightPos, 15, cv::Scalar(255, 0, 0), 2);
        }

        // 4. Update UI
        updateVideoLabel(displayFrame);
    }

    void mousePressEvent(QMouseEvent *event) override {
        if (m_calibPoints.size() < 4) {
            // Map click from label coordinates to original camera coordinates
            QLabel *label = qobject_cast<QLabel*>(sender() ? sender() : m_videoLabel);
            if(!label) label = m_videoLabel;
            
            QPixmap pix = label->pixmap();
            if(pix.isNull()) return;

            float scaleX = (float)m_lastFrameSize.width / pix.width();
            float scaleY = (float)m_lastFrameSize.height / pix.height();

            cv::Point2f camPos(event->x() * scaleX, event->y() * scaleY);
            m_calibPoints.push_back(camPos);

            if (m_calibPoints.size() == 4) {
                calculateHomography();
                m_statusLabel->setText("Status: CALIBRATED! Point your lightgun at the TV. Press SPACE to shoot, 'R' to reset.");
            } else {
                m_statusLabel->setText(QString("Status: Click corner %1...").arg(m_calibPoints.size() + 1));
            }
        }
    }

    void keyPressEvent(QKeyEvent *event) override {
        if (event->key() == Qt::Key_R) {
            m_calibPoints.clear();
            m_isCalibrated = false;
            m_statusLabel->setText("Status: Calibration reset. Click the 4 corners of your TV...");
        } 
        else if (event->key() == Qt::Key_Space && m_isCalibrated) {
            simulateMouseClick();
        }
        QWidget::keyPressEvent(event);
    }

private slots:
    void handleError(const QString &msg) {
        QMessageBox::critical(this, "Camera Error", msg);
        QApplication::quit();
    }

private:
    void updateVideoLabel(const cv::Mat &frame) {
        // Convert OpenCV Mat to QImage
        QImage img(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
        img = img.rgbSwapped(); // OpenCV is BGR, Qt is RGB
        
        m_lastFrameSize = cv::Size(frame.cols, frame.rows);
        m_videoLabel->setPixmap(QPixmap::fromImage(img).scaled(
            m_videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    void calculateHomography() {
        // Destination points are the corners of the primary screen
        std::vector<cv::Point2f> dst_pts = {
            cv::Point2f(0, 0),
            cv::Point2f(m_screenWidth, 0),
            cv::Point2f(m_screenWidth, m_screenHeight),
            cv::Point2f(0, m_screenHeight)
        };

        m_homography = cv::getPerspectiveTransform(m_calibPoints, dst_pts);
        m_isCalibrated = true;
    }

    void simulateMouseClick() {
#ifdef Q_OS_WIN
        // Windows API for mouse click
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
        
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
#else
        // Fallback for Linux/Mac (Requires XTest on Linux, or CGEvent on Mac)
        // For this single-file demo, we just print to console on non-Windows
        qDebug() << "FIRE! (Implement platform-specific click for Linux/Mac)";
#endif
    }

    QLabel *m_videoLabel;
    QLabel *m_statusLabel;
    QThread *m_cameraThread;
    CameraWorker *m_cameraWorker;

    std::vector<cv::Point2f> m_calibPoints;
    cv::Mat m_homography;
    bool m_isCalibrated = false;
    cv::Size m_lastFrameSize;
    
    int m_screenWidth;
    int m_screenHeight;
};

// ---------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    LightgunWidget window;
    window.show();

    return app.exec();
}

#include "main.moc"