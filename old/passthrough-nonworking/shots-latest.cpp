#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QThread>
#include <QKeyEvent>
#include <QCursor>
#include <QScreen>
#include <QPainter>
#include <QPushButton>
#include <QTimer>
#include <QMessageBox>
#include <QDateTime>
#include <QDebug>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// ---------------------------------------------------------
// 1. The Screen Overlay (Non-passthrough, shows shots)
// ---------------------------------------------------------
struct ShotImpact {
    QPoint pos;
    qint64 time;
};

class GameScreenOverlay : public QWidget {
    Q_OBJECT
public:
    GameScreenOverlay(QWidget *parent = nullptr) : QWidget(parent) {
        // Frameless, always on top. NO WA_TransparentForMouseEvents (it blocks clicks)
        setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint | Qt::Tool);
        
        QScreen *screen = QGuiApplication::primaryScreen();
        setGeometry(screen->geometry());
        
        m_cleanupTimer = new QTimer(this);
        connect(m_cleanupTimer, &QTimer::timeout, this, &GameScreenOverlay::cleanupImpacts);
        m_cleanupTimer->start(100);
    }

    void setCrosshairPos(const QPoint &pos) {
        m_crosshairPos = pos;
        update();
    }

    void addShotImpact(const QPoint &pos) {
        m_impacts.append({pos, QDateTime::currentMSecsSinceEpoch()});
        update();
    }

    void clearImpacts() {
        m_impacts.clear();
        update();
    }

    void flashBorder() {
        m_isFlashing = true;
        update();
        QTimer::singleShot(50, this, [this]() { m_isFlashing = false; update(); });
    }

    bool isTracking() const { return m_isTracking; }
    void setTracking(bool tracking) { m_isTracking = tracking; }

protected:
    void paintEvent(QPaintEvent *) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        
        // 1. Draw Border
        QColor borderColor = m_isFlashing ? Qt::white : QColor(0, 255, 0); 
        QPen pen(borderColor, 25); 
        p.setPen(pen);
        p.drawRect(rect().adjusted(12, 12, -12, -12)); 

        // 2. Draw Shot Impacts (Red bullet holes)
        p.setPen(QPen(Qt::red, 4));
        for (const auto &impact : m_impacts) {
            p.drawEllipse(impact.pos, 15, 15);
            p.drawLine(impact.pos.x() - 20, impact.pos.y(), impact.pos.x() + 20, impact.pos.y());
            p.drawLine(impact.pos.x(), impact.pos.y() - 20, impact.pos.x(), impact.pos.y() + 20);
        }

        // 3. Draw Crosshair (Cyan)
        if (m_isTracking) {
            p.setPen(QPen(Qt::cyan, 3));
            int size = 25;
            p.drawLine(m_crosshairPos.x() - size, m_crosshairPos.y(), m_crosshairPos.x() + size, m_crosshairPos.y());
            p.drawLine(m_crosshairPos.x(), m_crosshairPos.y() - size, m_crosshairPos.x(), m_crosshairPos.y() + size);
            p.drawEllipse(m_crosshairPos, 10, 10);
        }
    }

private slots:
    void cleanupImpacts() {
        qint64 now = QDateTime::currentMSecsSinceEpoch();
        bool changed = false;
        for (int i = m_impacts.size() - 1; i >= 0; --i) {
            if (now - m_impacts[i].time > 2000) { // Remove after 2 seconds
                m_impacts.removeAt(i);
                changed = true;
            }
        }
        if (changed) update();
    }

private:
    QPoint m_crosshairPos;
    bool m_isTracking = false;
    bool m_isFlashing = false;
    QList<ShotImpact> m_impacts;
    QTimer *m_cleanupTimer;
};

// ---------------------------------------------------------
// 2. Camera Worker Thread (FIXED FOR WEBCAM COMPATIBILITY)
// ---------------------------------------------------------
class CameraWorker : public QObject {
    Q_OBJECT
public:
    CameraWorker(QObject *parent = nullptr) : QObject(parent) {}

public slots:
    void startCamera() {
        cv::VideoCapture cap;
#ifdef Q_OS_WIN
        // DirectShow is much more reliable on Windows than the default MSMF
        cap.open(0, cv::CAP_DSHOW);
#else
        cap.open(0);
#endif
        
        if (!cap.isOpened()) {
            emit errorOccurred("Failed to open camera! Check Windows Privacy settings for Camera access.");
            return;
        }
        
        // FIX: DO NOT force resolution. Let the camera use its native default.
        // Forcing 1280x720 causes many webcams to fail and return empty frames.
        qDebug() << "Camera opened. Native resolution:" 
                 << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        cv::Mat frame;
        int emptyCount = 0;
        while (m_running) {
            cap >> frame;
            
            if (frame.empty()) {
                emptyCount++;
                if (emptyCount > 30) {
                    emit errorOccurred("Camera opened but returning empty frames. Try a different USB port.");
                    return;
                }
                QThread::msleep(50); 
                continue;
            }
            emptyCount = 0;

            if (frame.channels() == 1) {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            }

            emit frameReady(frame.clone());
            QThread::msleep(10); // ~100 FPS
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
// 3. Main Control Widget
// ---------------------------------------------------------
class SindenEmulator : public QWidget {
    Q_OBJECT
public:
    SindenEmulator(QWidget *parent = nullptr) : QWidget(parent) {
        setWindowTitle("Sinden Emulator - Control Panel");
        resize(400, 350);

        QVBoxLayout *layout = new QVBoxLayout(this);
        
        m_videoLabel = new QLabel("Waiting for camera...", this);
        m_videoLabel->setAlignment(Qt::AlignCenter);
        m_videoLabel->setStyleSheet("background-color: black; color: white; border: 1px solid gray;");
        m_videoLabel->setScaledContents(true); 
        layout->addWidget(m_videoLabel);

        m_statusLabel = new QLabel("Status: Initializing...", this);
        m_statusLabel->setStyleSheet("color: white; font-weight: bold; padding: 5px;");
        layout->addWidget(m_statusLabel);

        QHBoxLayout *btnLayout = new QHBoxLayout();
        QPushButton *btnToggleBorder = new QPushButton("Toggle Screen Overlay", this);
        connect(btnToggleBorder, &QPushButton::clicked, this, &SindenEmulator::toggleBorder);
        btnLayout->addWidget(btnToggleBorder);

        QPushButton *btnClearShots = new QPushButton("Clear Shot Marks", this);
        connect(btnClearShots, &QPushButton::clicked, this, &SindenEmulator::clearShots);
        btnLayout->addWidget(btnClearShots);
        layout->addLayout(btnLayout);

        m_cameraThread = new QThread(this);
        m_cameraWorker = new CameraWorker();
        m_cameraWorker->moveToThread(m_cameraThread);

        connect(m_cameraThread, &QThread::started, m_cameraWorker, &CameraWorker::startCamera);
        connect(m_cameraThread, &QThread::finished, m_cameraWorker, &QObject::deleteLater);
        connect(m_cameraWorker, &CameraWorker::frameReady, this, &SindenEmulator::processFrame);
        connect(m_cameraWorker, &CameraWorker::errorOccurred, this, &SindenEmulator::handleError);

        m_cameraThread->start();

        QScreen *screen = QGuiApplication::primaryScreen();
        m_screenWidth = screen->size().width();
        m_screenHeight = screen->size().height();

        m_overlay = new GameScreenOverlay();
        m_overlay->show();
    }

    ~SindenEmulator() {
        m_cameraWorker->stop();
        m_cameraThread->quit();
        m_cameraThread->wait();
        delete m_overlay;
    }

protected:
    void processFrame(const cv::Mat &frame) {
        cv::Mat displayFrame = frame.clone();
        cv::Mat hsv, mask;
        
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar lower_green(35, 50, 50);
        cv::Scalar upper_green(85, 255, 255);
        cv::inRange(hsv, lower_green, upper_green, mask);

        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point2f> screenCorners;
        double maxArea = 0;

        for (const auto &contour : contours) {
            double area = cv::contourArea(contour);
            if (area > maxArea && area > 5000) { 
                maxArea = area;
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);
                
                if (approx.size() == 4) {
                    screenCorners = orderPoints(approx);
                }
            }
        }

        if (screenCorners.size() == 4) {
            for (int i = 0; i < 4; ++i) {
                cv::line(displayFrame, screenCorners[i], screenCorners[(i+1)%4], cv::Scalar(255, 0, 255), 3);
            }

            std::vector<cv::Point2f> dst_pts = {
                cv::Point2f(0, 0),
                cv::Point2f(m_screenWidth, 0),
                cv::Point2f(m_screenWidth, m_screenHeight),
                cv::Point2f(0, m_screenHeight)
            };
            cv::Mat H = cv::getPerspectiveTransform(screenCorners, dst_pts);

            cv::Point2f cameraCenter(frame.cols / 2.0f, frame.rows / 2.0f);
            std::vector<cv::Point2f> src_pts = {cameraCenter};
            std::vector<cv::Point2f> dst_aim;
            cv::perspectiveTransform(src_pts, dst_aim, H);

            int targetX = static_cast<int>(dst_aim[0].x);
            int targetY = static_cast<int>(dst_aim[0].y);

            targetX = std::max(0, std::min(m_screenWidth - 1, targetX));
            targetY = std::max(0, std::min(m_screenHeight - 1, targetY));
            
            // Update the overlay crosshair instead of OS mouse
            m_overlay->setCrosshairPos(QPoint(targetX, targetY));
            m_overlay->setTracking(true);
            m_lastAimPos = QPoint(targetX, targetY);

            cv::drawMarker(displayFrame, cameraCenter, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 30, 3);
            m_statusLabel->setText("Status: TRACKING BORDER. Press SPACE to shoot.");
        } else {
            m_overlay->setTracking(false);
            m_statusLabel->setText("Status: NO BORDER DETECTED. Adjust camera or lighting.");
        }

        updateVideoLabel(displayFrame);
    }

    void keyPressEvent(QKeyEvent *event) override {
        if (event->key() == Qt::Key_Space) {
            if (m_overlay->isTracking()) {
                m_overlay->addShotImpact(m_lastAimPos);
                m_overlay->flashBorder();
            }
        }
        QWidget::keyPressEvent(event);
    }

private slots:
    void toggleBorder() {
        if (m_overlay->isVisible()) {
            m_overlay->hide();
        } else {
            m_overlay->show();
        }
    }

    void clearShots() {
        m_overlay->clearImpacts();
    }

    void handleError(const QString &msg) {
        QMessageBox::critical(this, "Camera Error", msg);
        QApplication::quit();
    }

private:
    void updateVideoLabel(const cv::Mat &frame) {
        m_videoLabel->setText(""); 
        
        QImage img;
        if (frame.channels() == 3) {
            img = QImage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888).rgbSwapped();
        } else if (frame.channels() == 1) {
            img = QImage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_Grayscale8);
        }
        
        m_videoLabel->setPixmap(QPixmap::fromImage(img));
    }

    std::vector<cv::Point2f> orderPoints(std::vector<cv::Point> pts) {
        std::vector<cv::Point2f> rect(4);
        cv::Point2f tl, tr, br, bl;

        std::vector<int> s;
        for (auto p : pts) s.push_back(p.x + p.y);
        tl = pts[std::distance(s.begin(), std::min_element(s.begin(), s.end()))];
        br = pts[std::distance(s.begin(), std::max_element(s.begin(), s.end()))];

        std::vector<int> d;
        for (auto p : pts) d.push_back(p.y - p.x);
        tr = pts[std::distance(d.begin(), std::min_element(d.begin(), d.end()))];
        bl = pts[std::distance(d.begin(), std::max_element(d.begin(), d.end()))];

        return {tl, tr, br, bl};
    }

    QLabel *m_videoLabel;
    QLabel *m_statusLabel;
    QThread *m_cameraThread;
    CameraWorker *m_cameraWorker;
    GameScreenOverlay *m_overlay;
    
    int m_screenWidth;
    int m_screenHeight;
    QPoint m_lastAimPos;
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