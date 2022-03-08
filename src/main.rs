use anyhow::Result;
use opencv::{core, highgui, imgcodecs, imgproc, prelude::*, videoio};

fn main() -> Result<()> {
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;
    let mut cam = videoio::VideoCapture::from_file("950_Video_Trim.mp4", videoio::CAP_ANY)?;
    let mut frame = Mat::default();

    let template = imgcodecs::imread("162.jpg", 0)?;

    println!(
        "{} {} {}",
        template.rows(),
        template.cols(),
        template.dims()
    );

    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            let mut gray = Mat::default();
            imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
            let mut gray_thresh = Mat::default();
            imgproc::threshold(&gray, &mut gray_thresh, 237.0, 255.0, 0)?;

            // template matching
            let mut match_result = Mat::default();
            imgproc::match_template(
                &gray_thresh,
                &template,
                &mut match_result,
                imgproc::TemplateMatchModes::TM_CCOEFF_NORMED as i32,
                &core::no_array(),
            )?;

            let mut max_val: f64 = 0.0;

            let mut max_loc = core::Point::new(0, 0);

            core::min_max_loc(
                &match_result,
                None,
                Some(&mut max_val),
                None,
                Some(&mut max_loc),
                &core::no_array(),
            )?;

            println!("{} {}", max_loc.x, max_loc.y);

            highgui::imshow("window", &gray_thresh)?;
        }

        let key = highgui::wait_key(1)?;
        // quit with q key
        if key == 113 {
            break;
        }
    }
    Ok(())
}
