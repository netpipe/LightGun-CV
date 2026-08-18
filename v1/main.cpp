// main.cpp
//
// Qt 5.12 + OpenCV single-file demo:
// Sinden-style lightgun emulation using a green border for calibration.
//
// Build example on Linux:
//   g++ -std=c++11 -fPIC main.cpp $(pkg-config --cflags --libs Qt5Widgets opencv4) -o green-lightgun
//
// On Windows/MinGW you may need to add:
//   -luser32
//

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QSpinBox>
#include <QSlider>
#include <QGroupBox>
#include <QFormLayout>
#include <QTimer>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QPen>
#include <QCursor>
#include <QKeyEvent>
#include <QTimerEvent>
#include <QStatusBar>
#include <QProcess>
#include <QStringList>
#include <QShortcut>
#include <QKeySequence>
#include <QtGlobal>
#include <QtMath>
#include <QVector>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef Q_OS_WIN
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#ifndef WINVER
#define WINVER 0x0600
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
#pragma comment(lib, "user32.lib")
#endif

struct Params
{
    int hMin = 35;      // OpenCV hue range is 0..179
    int hMax = 85;
    int sMin = 70;
    int vMin = 60;
    int minArea = 1500;
};

static std::vector<cv::Point2f> orderQuad(const std::vector<cv::Point2f> &pts)
{
    if (pts.size() != 4)
        return pts;

    std::vector<cv::Point2f> out(4);

    float sums[4];
    float diffs[4];

    for (int i = 0; i < 4; ++i) {
        sums[i]  = pts[i].x + pts[i].y;
        diffs[i] = pts[i].y - pts[i].x;
    }

    int tl = 0, tr = 0, br = 0, bl = 0;

    for (int i = 1; i < 4; ++i) {
        if (sums[i]  < sums[tl]) tl = i;
        if (sums[i]  > sums[br]) br = i;
        if (diffs[i] < diffs[tr]) tr = i;
        if (diffs[i] > diffs[bl]) bl = i;
    }

    // Order: TL, TR, BR, BL
    out[0] = pts[tl];
    out[1] = pts[tr];
    out[2] = pts[br];
    out[3] = pts[bl];

    return out;
}

static bool detectGreenBorder(const cv::Mat &bgr,
                               const Params &p,
                               std::vector<cv::Point2f> &quad,
                               double &area)
{
    if (bgr.empty())
        return false;

    cv::Mat hsv;
    cv::Mat mask;

    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    if (p.hMin <= p.hMax) {
        cv::inRange(hsv,
                    cv::Scalar(p.hMin, p.sMin, p.vMin),
                    cv::Scalar(p.hMax, 255, 255),
                    mask);
    } else {
        cv::Mat m1, m2;
        cv::inRange(hsv,
                    cv::Scalar(p.hMin, p.sMin, p.vMin),
                    cv::Scalar(179, 255, 255),
                    m1);
        cv::inRange(hsv,
                    cv::Scalar(0, p.sMin, p.vMin),
                    cv::Scalar(p.hMax, 255, 255),
                    m2);
        cv::bitwise_or(m1, m2, mask);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return false;

    int best = -1;
    double bestArea = 0.0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > bestArea) {
            bestArea = a;
            best = static_cast<int>(i);
        }
    }

    if (best < 0 || bestArea < p.minArea)
        return false;

    area = bestArea;

    double peri = cv::arcLength(contours[best], true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contours[best], approx, 0.02 * peri, true);

    std::vector<cv::Point2f> pts;

    if (approx.size() == 4 && cv::isContourConvex(approx)) {
        for (const auto &pt : approx)
            pts.push_back(pt);
    } else {
        cv::RotatedRect rr = cv::minAreaRect(contours[best]);
        cv::Point2f tmp[4];
        rr.points(tmp);

        for (int i = 0; i < 4; ++i)
            pts.push_back(tmp[i]);
    }

    if (pts.size() != 4)
        return false;

    quad = orderQuad(pts);
    return true;
}

class BorderWindow : public QWidget
{
public:
    explicit BorderWindow(QWidget *parent = nullptr)
        : QWidget(parent)
    {
        setWindowTitle("Green Border");
        resize(800, 480);
        setMinimumSize(320, 240);
    }

    void setThickness(int t)
    {
        thickness = qBound(4, t, 200);
        update();
    }

    void showWindowed()
    {
        showNormal();
        raise();
        activateWindow();
    }

    void addShot(const QPoint &globalPos)
    {
        shots.append(clampedLocal(globalPos));

        if (shots.size() > 30)
            shots.removeFirst();

        if (shotTimer == 0)
            shotTimer = startTimer(400);

        update();
    }

    void setAim(const QPoint &globalPos)
    {
        aim = clampedLocal(globalPos);
        hasAim = true;
        update();
    }

    void clearAim()
    {
        hasAim = false;
        update();
    }

    std::vector<cv::Point2f> screenCorners() const
    {
        QPoint tl = mapToGlobal(QPoint(0, 0));
        QPoint tr = mapToGlobal(QPoint(width() - 1, 0));
        QPoint br = mapToGlobal(QPoint(width() - 1, height() - 1));
        QPoint bl = mapToGlobal(QPoint(0, height() - 1));

        std::vector<cv::Point2f> v;
        v.reserve(4);

        // Must match orderQuad(): TL, TR, BR, BL
        v.push_back(cv::Point2f(static_cast<float>(tl.x()), static_cast<float>(tl.y())));
        v.push_back(cv::Point2f(static_cast<float>(tr.x()), static_cast<float>(tr.y())));
        v.push_back(cv::Point2f(static_cast<float>(br.x()), static_cast<float>(br.y())));
        v.push_back(cv::Point2f(static_cast<float>(bl.x()), static_cast<float>(bl.y())));

        return v;
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);

        // Pure green outer border.
        p.fillRect(rect(), QColor(0, 255, 0));

        // Black inside.
        QRect inner = rect().adjusted(thickness, thickness, -thickness, -thickness);
        if (inner.width() > 0 && inner.height() > 0)
            p.fillRect(inner, Qt::black);

        if (hasAim) {
            p.setPen(QPen(QColor(255, 64, 64), 2));
            p.drawLine(aim.x() - 16, aim.y(), aim.x() + 16, aim.y());
            p.drawLine(aim.x(), aim.y() - 16, aim.x(), aim.y() + 16);
        }

        p.setPen(QPen(Qt::red, 3));
        for (const QPoint &s : shots) {
            p.drawEllipse(s, 13, 13);
            p.drawLine(s.x() - 18, s.y(), s.x() + 18, s.y());
            p.drawLine(s.x(), s.y() - 18, s.x(), s.y() + 18);
        }
    }

    void keyPressEvent(QKeyEvent *e) override
    {
        if (e->key() == Qt::Key_Escape) {
            hide();
            e->accept();
            return;
        }

        QWidget::keyPressEvent(e);
    }

    void timerEvent(QTimerEvent *e) override
    {
        Q_UNUSED(e);

        shots.clear();

        if (shotTimer != 0) {
            killTimer(shotTimer);
            shotTimer = 0;
        }

        update();
    }

private:
    QPoint clampedLocal(const QPoint &globalPos) const
    {
        QPoint p = mapFromGlobal(globalPos);
        int maxX = qMax(0, width() - 1);
        int maxY = qMax(0, height() - 1);

        return QPoint(qBound(0, p.x(), maxX),
                      qBound(0, p.y(), maxY));
    }

    int thickness = 24;
    int shotTimer = 0;
    QVector<QPoint> shots;
    QPoint aim;
    bool hasAim = false;
};

static void clickAt(const QPoint &globalPos)
{
#ifdef Q_OS_WIN
    SetCursorPos(globalPos.x(), globalPos.y());

    INPUT inputs[2];
    std::memset(inputs, 0, sizeof(inputs));

    inputs[0].type = INPUT_MOUSE;
    inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

    inputs[1].type = INPUT_MOUSE;
    inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;

    SendInput(2, inputs, sizeof(INPUT));
#elif defined(Q_OS_LINUX)
    // Optional Linux click support.
    // Requires xdotool to be installed.
    QProcess::startDetached("xdotool",
                            QStringList()
                            << "mousemove"
                            << QString::number(globalPos.x())
                            << QString::number(globalPos.y())
                            << "click"
                            << "1");
#else
    QCursor::setPos(globalPos);
#endif
}

class MainWindow : public QMainWindow
{
public:
    MainWindow()
    {
        border = new BorderWindow();

        QWidget *central = new QWidget();
        QVBoxLayout *root = new QVBoxLayout(central);

        videoLabel = new QLabel("Camera view");
        videoLabel->setMinimumSize(640, 480);
        videoLabel->setAlignment(Qt::AlignCenter);
        videoLabel->setStyleSheet("background:#101010; color:#ddd; border:1px solid #333;");
        root->addWidget(videoLabel, 1);

        QHBoxLayout *btnRow = new QHBoxLayout();

        btnStart = new QPushButton("Start camera");
        btnStop = new QPushButton("Stop camera");
        btnCalibrate = new QPushButton("Calibrate (C)");
        btnShoot = new QPushButton("Shoot (Space)");
        btnBorder = new QPushButton("Toggle border (B)");
        btnFullBorder = new QPushButton("Fullscreen border (F)");

        btnRow->addWidget(btnStart);
        btnRow->addWidget(btnStop);
        btnRow->addWidget(btnCalibrate);
        btnRow->addWidget(btnShoot);
        btnRow->addWidget(btnBorder);
        btnRow->addWidget(btnFullBorder);
        btnRow->addStretch();

        root->addLayout(btnRow);

        for (QPushButton *b : {btnStart, btnStop, btnCalibrate, btnShoot, btnBorder, btnFullBorder})
            b->setFocusPolicy(Qt::NoFocus);

        QGroupBox *group = new QGroupBox("Settings");
        QFormLayout *form = new QFormLayout(group);

        spinCamera = new QSpinBox();
        spinCamera->setRange(0, 16);
        spinCamera->setValue(0);

        spinThickness = new QSpinBox();
        spinThickness->setRange(4, 200);
        spinThickness->setValue(24);

        spinMinArea = new QSpinBox();
        spinMinArea->setRange(100, 500000);
        spinMinArea->setValue(1500);
        spinMinArea->setSingleStep(100);

        slHMin = new QSlider(Qt::Horizontal);
        slHMin->setRange(0, 179);
        slHMin->setValue(35);

        slHMax = new QSlider(Qt::Horizontal);
        slHMax->setRange(0, 179);
        slHMax->setValue(85);

        slSat = new QSlider(Qt::Horizontal);
        slSat->setRange(0, 255);
        slSat->setValue(70);

        slVal = new QSlider(Qt::Horizontal);
        slVal->setRange(0, 255);
        slVal->setValue(60);

        chkMove = new QCheckBox("Move system cursor to aim point");
        chkMove->setChecked(true);

        chkClick = new QCheckBox("Send OS left click on shoot");
        chkClick->setChecked(true);

        chkAuto = new QCheckBox("Auto-recalibrate while border visible");
        chkAuto->setChecked(false);

        chkMirror = new QCheckBox("Mirror camera horizontally");
        chkMirror->setChecked(false);

        form->addRow("Camera index", spinCamera);
        form->addRow("Border thickness", spinThickness);
        form->addRow("Min border area (px)", spinMinArea);
        form->addRow("Hue min", slHMin);
        form->addRow("Hue max", slHMax);
        form->addRow("Saturation min", slSat);
        form->addRow("Value min", slVal);
        form->addRow(QString(), chkMove);
        form->addRow(QString(), chkClick);
        form->addRow(QString(), chkAuto);
        form->addRow(QString(), chkMirror);

        root->addWidget(group);

        setCentralWidget(central);
        setWindowTitle("Qt 5.12 OpenCV Green Border Lightgun Demo");

        statusBar()->showMessage(
            "Ready. Show the green border, start the camera, then press Calibrate.");

        border->setThickness(spinThickness->value());

        connect(btnStart, &QPushButton::clicked, [this]() {
            startCamera();
        });

        connect(btnStop, &QPushButton::clicked, [this]() {
            stopCamera();
        });

        connect(btnCalibrate, &QPushButton::clicked, [this]() {
            calibrate();
        });

        connect(btnShoot, &QPushButton::clicked, [this]() {
            shoot();
        });

        connect(btnBorder, &QPushButton::clicked, [this]() {
            toggleBorder();
        });

        connect(btnFullBorder, &QPushButton::clicked, [this]() {
            border->showFullScreen();
            border->raise();
            border->activateWindow();
        });

        connect(spinThickness, QOverload<int>::of(&QSpinBox::valueChanged), [this](int v) {
            border->setThickness(v);
        });

        timer.setInterval(33);
        connect(&timer, &QTimer::timeout, [this]() {
            grabFrame();
        });

        QShortcut *scShoot = new QShortcut(QKeySequence(Qt::Key_Space), this);
        scShoot->setContext(Qt::ApplicationShortcut);
        connect(scShoot, &QShortcut::activated, [this]() {
            shoot();
        });

        QShortcut *scCal = new QShortcut(QKeySequence(Qt::Key_C), this);
        scCal->setContext(Qt::ApplicationShortcut);
        connect(scCal, &QShortcut::activated, [this]() {
            calibrate();
        });

        QShortcut *scBorder = new QShortcut(QKeySequence(Qt::Key_B), this);
        scBorder->setContext(Qt::ApplicationShortcut);
        connect(scBorder, &QShortcut::activated, [this]() {
            toggleBorder();
        });

        QShortcut *scFull = new QShortcut(QKeySequence(Qt::Key_F), this);
        scFull->setContext(Qt::ApplicationShortcut);
        connect(scFull, &QShortcut::activated, [this]() {
            border->showFullScreen();
            border->raise();
            border->activateWindow();
        });

        updateButtons();
        resize(880, 880);
    }

    ~MainWindow()
    {
        stopCamera();

        if (border) {
            border->close();
            delete border;
            border = nullptr;
        }
    }

private:
    Params currentParams() const
    {
        Params p;
        p.hMin = slHMin->value();
        p.hMax = slHMax->value();
        p.sMin = slSat->value();
        p.vMin = slVal->value();
        p.minArea = spinMinArea->value();
        return p;
    }

    bool startCamera()
    {
        if (running)
            return true;

        int idx = spinCamera->value();

        if (!cap.open(idx)) {
#ifdef Q_OS_WIN
            cap.release();
            cap.open(idx, cv::CAP_DSHOW);
#endif
        }

        if (!cap.isOpened()) {
            statusBar()->showMessage("Failed to open camera.", 3000);
            return false;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        running = true;
        timer.start();

        updateButtons();
        statusBar()->showMessage("Camera started.", 2000);

        return true;
    }

    void stopCamera()
    {
        timer.stop();

        if (cap.isOpened())
            cap.release();

        running = false;
        haveAim = false;

        updateButtons();
        statusBar()->showMessage("Camera stopped.", 1500);
    }

    void updateButtons()
    {
        btnStart->setEnabled(!running);
        btnStop->setEnabled(running);
    }

    void toggleBorder()
    {
        if (border->isVisible())
            border->hide();
        else
            border->showWindowed();
    }

    void calibrate()
    {
        if (!border->isVisible())
            border->showWindowed();

        if (!running && !startCamera())
            return;

        pendingCalibration = true;
        pendingTries = 45;

        statusBar()->showMessage(
            "Calibrating: make sure the whole green border is visible to the camera.",
            3000);
    }

    void updateCalibration(const std::vector<cv::Point2f> &camQuad, bool verbose)
    {
        if (!border->isVisible()) {
            if (verbose)
                statusBar()->showMessage("Show the green border first.", 3000);
            return;
        }

        std::vector<cv::Point2f> screenQuad = border->screenCorners();

        cv::Mat h = cv::findHomography(camQuad, screenQuad, 0);

        if (!h.empty()) {
            homography = h;
            calibrated = true;

            if (verbose)
                statusBar()->showMessage("Calibration OK.", 2000);
        } else {
            if (verbose)
                statusBar()->showMessage("Homography failed. Try again.", 3000);
        }
    }

    void grabFrame()
    {
        if (!running)
            return;

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty())
            return;

        if (chkMirror && chkMirror->isChecked())
            cv::flip(frame, frame, 1);

        lastFrame = frame.clone();

        Params p = currentParams();

        std::vector<cv::Point2f> quad;
        double area = 0.0;

        bool found = detectGreenBorder(frame, p, quad, area);

        cv::Mat display = frame.clone();

        if (found) {
            for (int i = 0; i < 4; ++i) {
                cv::line(display, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                cv::circle(display, quad[i], 4, cv::Scalar(0, 0, 255), -1);
            }

            std::string areaText = "border area: " + std::to_string(static_cast<int>(area));
            cv::putText(display,
                        areaText,
                        cv::Point(10, display.rows - 40),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(255, 255, 255),
                        1);
        }

        if (pendingCalibration) {
            if (found) {
                updateCalibration(quad, true);
                pendingCalibration = false;
                pendingTries = 0;
            } else if (--pendingTries <= 0) {
                pendingCalibration = false;
                statusBar()->showMessage(
                    "Calibration failed: no green border detected. Adjust sliders.",
                    3000);
            }
        } else if (found && chkAuto && chkAuto->isChecked() && border->isVisible()) {
            if (++autoCounter % 12 == 0)
                updateCalibration(quad, false);
        }

        if (calibrated && !homography.empty()) {
            cv::Point2f center(frame.cols / 2.0f, frame.rows / 2.0f);

            std::vector<cv::Point2f> in;
            std::vector<cv::Point2f> out;

            in.push_back(center);
            cv::perspectiveTransform(in, out, homography);

            if (!out.empty() &&
                std::isfinite(out[0].x) &&
                std::isfinite(out[0].y)) {
                haveAim = true;
                lastAim = QPoint(qRound(out[0].x), qRound(out[0].y));

                int cx = qRound(center.x);
                int cy = qRound(center.y);

                cv::line(display, cv::Point(cx - 18, cy), cv::Point(cx + 18, cy),
                         cv::Scalar(0, 255, 255), 2);
                cv::line(display, cv::Point(cx, cy - 18), cv::Point(cx, cy + 18),
                         cv::Scalar(0, 255, 255), 2);

                std::string aimText =
                    "Aim: " + std::to_string(lastAim.x()) + "," + std::to_string(lastAim.y());

                cv::putText(display,
                            aimText,
                            cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.6,
                            cv::Scalar(0, 255, 255),
                            1);

                if (chkMove && chkMove->isChecked()) {
                    QCursor::setPos(lastAim);

                    if (border->isVisible())
                        border->setAim(lastAim);
                }
            } else {
                haveAim = false;
            }
        } else {
            cv::putText(display,
                        "Not calibrated",
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(0, 0, 255),
                        2);
        }

        cv::Mat rgb;
        cv::cvtColor(display, rgb, cv::COLOR_BGR2RGB);

        QImage img(rgb.data,
                   rgb.cols,
                   rgb.rows,
                   static_cast<int>(rgb.step),
                   QImage::Format_RGB888);

        img = img.copy();

        videoLabel->setPixmap(
            QPixmap::fromImage(img).scaled(videoLabel->size(),
                                           Qt::KeepAspectRatio,
                                           Qt::SmoothTransformation));
    }

    void shoot()
    {
        if (!calibrated || !haveAim) {
            statusBar()->showMessage("Calibrate first, or no valid aim point.", 2000);
            return;
        }

        border->addShot(lastAim);

        if (chkClick && chkClick->isChecked())
            clickAt(lastAim);
        else
            QCursor::setPos(lastAim);

        statusBar()->showMessage(
            QString("Shot at %1,%2").arg(lastAim.x()).arg(lastAim.y()),
            1500);
    }

private:
    BorderWindow *border = nullptr;

    QLabel *videoLabel = nullptr;

    QPushButton *btnStart = nullptr;
    QPushButton *btnStop = nullptr;
    QPushButton *btnCalibrate = nullptr;
    QPushButton *btnShoot = nullptr;
    QPushButton *btnBorder = nullptr;
    QPushButton *btnFullBorder = nullptr;

    QCheckBox *chkMove = nullptr;
    QCheckBox *chkClick = nullptr;
    QCheckBox *chkAuto = nullptr;
    QCheckBox *chkMirror = nullptr;

    QSpinBox *spinCamera = nullptr;
    QSpinBox *spinThickness = nullptr;
    QSpinBox *spinMinArea = nullptr;

    QSlider *slHMin = nullptr;
    QSlider *slHMax = nullptr;
    QSlider *slSat = nullptr;
    QSlider *slVal = nullptr;

    QTimer timer;
    cv::VideoCapture cap;

    bool running = false;

    cv::Mat lastFrame;
    cv::Mat homography;

    bool calibrated = false;
    bool haveAim = false;
    QPoint lastAim;

    bool pendingCalibration = false;
    int pendingTries = 0;

    int autoCounter = 0;
};

int main(int argc, char *argv[])
{
    // Keep screen pixel coordinates simple for lightgun mapping.
    QApplication::setAttribute(Qt::AA_DisableHighDpiScaling);

    QApplication app(argc, argv);
    app.setApplicationName("Green Border Lightgun Demo");

    MainWindow w;
    w.show();

    return app.exec();
}
