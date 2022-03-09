use anyhow::Result;
use opencv::{core, highgui, imgcodecs, imgproc, prelude::*, videoio};
use std::env;
// use rayon;
// use rayon::prelude::*;

fn get_video_path() -> Result<String> {
    let mut bin_dir = env::current_exe()?;
    bin_dir.pop();
    bin_dir.pop();
    bin_dir.pop();
    bin_dir.push("950_Video_Trim.mp4");
    Ok(bin_dir.display().to_string())
}

fn main() -> Result<()> {
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;
    let mut cam = videoio::VideoCapture::from_file(&get_video_path()?, videoio::CAP_ANY)?;
    let mut frame = Mat::default();

    // let template = imgcodecs::imread("162.jpg", 0)?;

    let templates: [Mat; 9] = [imgcodecs::imread("photos/68.jpg", 0)?, 
                                imgcodecs::imread("photos/68(2).jpg", 0)?, 
                                imgcodecs::imread("photos/74.5.jpg", 0)?, 
                                imgcodecs::imread("photos/84.jpg", 0)?,
                                imgcodecs::imread("photos/115.jpg", 0)?,
                                imgcodecs::imread("photos/162.jpg", 0)?,
                                imgcodecs::imread("photos/209.jpg", 0)?,
                                imgcodecs::imread("photos/256.jpg", 0)?,
                                imgcodecs::imread("photos/303.jpg", 0)?];

    // let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;

    // println!(
    //     "{} {} {}",
    //     template.rows(),
    //     template.cols(),
    //     template.dims()
    // );

    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            let mut gray = Mat::default();
            imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
            let mut gray_thresh = Mat::default();
            imgproc::threshold(&gray, &mut gray_thresh, 237.0, 255.0, 0)?;
            
            let mut match_result = Mat::default();
            let mut highest_match: f64 = 0.0;

            // template matching
            for template in &templates {
                let mut temp_match_result = Mat::default();
                imgproc::match_template(
                    &gray_thresh,
                    &template,
                    &mut temp_match_result,
                    imgproc::TemplateMatchModes::TM_CCOEFF_NORMED as i32,
                    &core::no_array(),
                )?;

                let mut result_max: f64 = 0.0;

                core::min_max_loc(&temp_match_result, None, Some(&mut result_max), None, None, &core::no_array())?;

                if result_max > highest_match {
                    highest_match = result_max;
                    match_result = temp_match_result;
                }
            } 

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
