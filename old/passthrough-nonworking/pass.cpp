#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QThread>
#include <QKeyEvent>
#include <QCursor>
#include <QScreen>
#include <QPainter>
#include <QPushButton>
#include <QTimer>
#include <QMessageBox>
#include <QMouseEvent>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

// ---------------------------------------------------------
// 1. The Screen Border Overlay (Emulates Sinden Software)
// ---------------------------------------------------------
class BorderOverlay : public QWidget {
    Q_OBJECT
public:
    BorderOverlay(QWidget *parent = nullptr) : QWidget(parent) {
        // Make it fullscreen, frameless, and always on top
        setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint | Qt::Tool);
        setAttribute(Qt::WA_TranslucentBackground);
        setAttribute(Qt::WA_TransparentForMouseEvents); // Let clicks pass through to the game!

        QScreen *screen = QGuiApplication::primaryScreen();
        setGeometry(screen->geometry());
        
        m_flashTimer = new QTimer(this);
        connect(m_flashTimer, &QTimer::timeout, this, [this]() { m_isFlashing = false; update(); });
    }

    void flashBorder() {
        m_isFlashing = true;
        update();
        m_flashTimer->start(50); // Flash for 50ms
    }

protected:
    void paintEvent(QPaintEvent *) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        
        // Sinden uses a bright, specific color. Neon Green is easiest for OpenCV to isolate.
        QColor borderColor = m_isFlashing ? Qt::white : QColor(0, 255, 0); 
        QPen pen(borderColor, 25); // 25px thick border
        p.setPen(pen);
        
        // Draw the border slightly inset so it doesn't get cut off by screen edges
        p.drawRect(rect().adjusted(12, 12, -12, -12)); 
    }

private:
    bool m_isFlashing = false;
    QTimer *m_flashTimer;
};

// ---------------------------------------------------------
// 2. Camera Worker Thread
// ---------------------------------------------------------
class CameraWorker : public QObject {
    Q_OBJECT
public:
    CameraWorker(QObject *parent = nullptr) : QObject(parent) {}

public slots:
    void startCamera() {
        cv::VideoCapture cap(0); 
        if (!cap.isOpened()) {
            emit errorOccurred("Failed to open camera!");
            return;
        }
        
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        cv::Mat frame;
        while (m_running) {
            cap >> frame;
            if (frame.empty()) break;
            emit frameReady(frame.clone());
            QThread::msleep(5); // ~200 FPS cap
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
// 3. Main Sinden Emulator Widget
// ---------------------------------------------------------
class SindenEmulator : public QWidget {
    Q_OBJECT
public:
    SindenEmulator(QWidget *parent = nullptr) : QWidget(parent) {
        setWindowTitle("Sinden Lightgun Emulator (Qt/OpenCV)");
        resize(640, 480);

        QVBoxLayout *layout = new QVBoxLayout(this);
        m_videoLabel = new QLabel("Waiting for camera...", this);
        m_videoLabel->setAlignment(Qt::AlignCenter);
        m_videoLabel->setStyleSheet("background-color: black; color: white;");
        layout->addWidget(m_videoLabel);

        m_statusLabel = new QLabel("Status: Point webcam at the Green Border. Center of camera = Crosshair.", this);
        m_statusLabel->setStyleSheet("color: white; font-weight: bold; padding: 5px;");
        layout->addWidget(m_statusLabel);

        QPushButton *btnToggleBorder = new QPushButton("Toggle Border Overlay", this);
        connect(btnToggleBorder, &QPushButton::clicked, this, &SindenEmulator::toggleBorder);
        layout->addWidget(btnToggleBorder);

        // Setup Camera Thread
        m_cameraThread = new QThread(this);
        m_cameraWorker = new CameraWorker();
        m_cameraWorker->moveToThread(m_cameraThread);

        connect(m_cameraThread, &QThread::started, m_cameraWorker, &CameraWorker::startCamera);
        connect(m_cameraThread, &QThread::finished, m_cameraWorker, &QObject::deleteLater);
        connect(m_cameraWorker, &CameraWorker::frameReady, this, &SindenEmulator::processFrame);
        connect(m_cameraWorker, &CameraWorker::errorOccurred, this, &SindenEmulator::handleError);

        m_cameraThread->start();

        // Screen resolution
        QScreen *screen = QGuiApplication::primaryScreen();
        m_screenWidth = screen->size().width();
        m_screenHeight = screen->size().height();

        // Start the border overlay
        m_borderOverlay = new BorderOverlay();
        m_borderOverlay->show();
    }

    ~SindenEmulator() {
        m_cameraWorker->stop();
        m_cameraThread->quit();
        m_cameraThread->wait();
        delete m_borderOverlay;
    }

protected:
    void processFrame(const cv::Mat &frame) {
        cv::Mat displayFrame = frame.clone();
        cv::Mat hsv, mask;
        
        // 1. Isolate the Neon Green Border using HSV
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar lower_green(35, 80, 80);
        cv::Scalar upper_green(85, 255, 255);
        cv::inRange(hsv, lower_green, upper_green, mask);

        // Clean up noise
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        // 2. Find the Screen Corners
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point2f> screenCorners;
        double maxArea = 0;

        for (const auto &contour : contours) {
            double area = cv::contourArea(contour);
            if (area > maxArea && area > 10000) { // Must be a large contour (the screen)
                maxArea = area;
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);
                
                if (approx.size() == 4) {
                    screenCorners = orderPoints(approx);
                }
            }
        }

        // 3. Draw Debug & Calculate Aim
        if (screenCorners.size() == 4) {
            // Draw detected border
            for (int i = 0; i < 4; ++i) {
                cv::line(displayFrame, screenCorners[i], screenCorners[(i+1)%4], cv::Scalar(255, 0, 255), 3);
                cv::circle(displayFrame, screenCorners[i], 8, cv::Scalar(0, 255, 255), -1);
            }

            // Calculate Homography (Map camera view to screen coordinates)
            std::vector<cv::Point2f> dst_pts = {
                cv::Point2f(0, 0),
                cv::Point2f(m_screenWidth, 0),
                cv::Point2f(m_screenWidth, m_screenHeight),
                cv::Point2f(0, m_screenHeight)
            };
            cv::Mat H = cv::getPerspectiveTransform(screenCorners, dst_pts);

            // The "Crosshair" is the exact center of the camera frame
            cv::Point2f cameraCenter(frame.cols / 2.0f, frame.rows / 2.0f);
            std::vector<cv::Point2f> src_pts = {cameraCenter};
            std::vector<cv::Point2f> dst_aim;
            cv::perspectiveTransform(src_pts, dst_aim, H);

            int targetX = static_cast<int>(dst_aim[0].x);
            int targetY = static_cast<int>(dst_aim[0].y);

            // Clamp and Move Mouse
            targetX = std::max(0, std::min(m_screenWidth - 1, targetX));
            targetY = std::max(0, std::min(m_screenHeight - 1, targetY));
            
            QCursor::setPos(targetX, targetY);

            // Draw Crosshair on camera feed
            cv::drawMarker(displayFrame, cameraCenter, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 30, 3);
            cv::putText(displayFrame, "TRACKING BORDER", cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(displayFrame, "NO BORDER DETECTED", cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        // 4. Update UI
        updateVideoLabel(displayFrame);
    }

    void keyPressEvent(QKeyEvent *event) override {
        if (event->key() == Qt::Key_Space) {
            simulateMouseClick();
            m_borderOverlay->flashBorder(); // Sinden recoil flash!
        }
        QWidget::keyPressEvent(event);
    }

private slots:
    void toggleBorder() {
        if (m_borderOverlay->isVisible()) {
            m_borderOverlay->hide();
            m_statusLabel->setText("Status: Border Hidden. Click button to show.");
        } else {
            m_borderOverlay->show();
            m_statusLabel->setText("Status: Point webcam at the Green Border.");
        }
    }

    void handleError(const QString &msg) {
        QMessageBox::critical(this, "Camera Error", msg);
        QApplication::quit();
    }

private:
    void updateVideoLabel(const cv::Mat &frame) {
        QImage img(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
        img = img.rgbSwapped();
        m_videoLabel->setPixmap(QPixmap::fromImage(img).scaled(
            m_videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    // Reorders 4 points to: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    std::vector<cv::Point2f> orderPoints(std::vector<cv::Point> pts) {
        std::vector<cv::Point2f> rect(4);
        cv::Point2f tl, tr, br, bl;

        // Sum of x and y
        std::vector<int> s;
        for (auto p : pts) s.push_back(p.x + p.y);
        tl = pts[std::distance(s.begin(), std::min_element(s.begin(), s.end()))];
        br = pts[std::distance(s.begin(), std::max_element(s.begin(), s.end()))];

        // Difference of y and x
        std::vector<int> d;
        for (auto p : pts) d.push_back(p.y - p.x);
        tr = pts[std::distance(d.begin(), std::min_element(d.begin(), d.end()))];
        bl = pts[std::distance(d.begin(), std::max_element(d.begin(), d.end()))];

        return {tl, tr, br, bl};
    }

    void simulateMouseClick() {
#ifdef Q_OS_WIN
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
#else
        qDebug() << "FIRE! (Implement platform-specific click for Linux/Mac)";
#endif
    }

    QLabel *m_videoLabel;
    QLabel *m_statusLabel;
    QThread *m_cameraThread;
    CameraWorker *m_cameraWorker;
    BorderOverlay *m_borderOverlay;
    
    int m_screenWidth;
    int m_screenHeight;
};

// ---------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    SindenEmulator window;
    window.show();

    return app.exec();
}

#include "main.moc"